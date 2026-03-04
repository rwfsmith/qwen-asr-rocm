"""
Qwen3-ASR ONNX inference server — onnxruntime direct (no optimum, no torch)
Serves OpenAI-compatible /v1/audio/transcriptions.

Model files (andrewleech/qwen3-asr-0.6b-onnx-int8):
  encoder.onnx       mel [1,128,T] -> audio_features [1,N,1024]
  decoder_init.onnx  input_embeds + position_ids -> logits + present_keys + present_values
  decoder_step.onnx  input_embeds + pos + past_keys + past_values -> logits + present_*
  embed_tokens.bin   [vocab, 1024] float16 token embedding matrix

Inference pipeline (from andrewleech/qwen3-asr-onnx README):
  1. Log-mel spectrogram (Whisper params) via librosa — pure numpy, no torch
  2. encoder.onnx -> audio_features [1, N, 1024]
  3. Build prompt token IDs with N <|audio_pad|> placeholders
  4. Look up token embeddings from embed_tokens.bin
  5. Replace <|audio_pad|> embeddings with audio_features[0]
  6. decoder_init.onnx -> logits + KV cache
  7. Greedy decode with decoder_step.onnx until EOS
"""

from __future__ import annotations

import io
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_DIR      = os.environ.get("MODEL_DIR", "/app/models")
MODEL_NAME     = os.environ.get("MODEL_NAME", "andrewleech/qwen3-asr-0.6b-onnx-int8")
PROCESSOR_NAME = os.environ.get("PROCESSOR_NAME", "Qwen/Qwen3-ASR-0.6B")
HOST           = os.environ.get("HOST", "0.0.0.0")
PORT           = int(os.environ.get("PORT", "8001"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "448"))
TARGET_SR      = 16_000

# ── Special token IDs (from andrewleech/qwen3-asr-onnx src/prompt.py) ─────────
ENDOFTEXT_TOKEN_ID  = 151643   # <|endoftext|>
IM_START_TOKEN_ID   = 151644   # <|im_start|>
IM_END_TOKEN_ID     = 151645   # <|im_end|>  — EOS
AUDIO_START_TOKEN_ID = 151669  # <|audio_start|>
AUDIO_END_TOKEN_ID  = 151670   # <|audio_end|>
AUDIO_PAD_TOKEN_ID  = 151676   # <|audio_pad|> — replaced by encoder output
EOS_TOKEN_IDS       = frozenset({ENDOFTEXT_TOKEN_ID, IM_END_TOKEN_ID})

# Fixed sub-word token IDs in Qwen3 tokenizer
_SYSTEM_TOKEN  = 9125   # "system"
_NEWLINE_TOKEN = 198    # "\n"

# Encoder conv parameters (0.6B model)
_CONV_WINDOW       = 100
_TOKENS_PER_WINDOW = 13


# ── Mel spectrogram (Whisper-compatible, pure numpy/librosa) ───────────────────
# Parameters: sr=16000, n_fft=400, hop=160, n_mels=128, fmin=0, fmax=8000
# Slaney-normalized filterbank, log10, global-max normalization.

_mel_filters: np.ndarray | None = None


def _get_mel_filters() -> np.ndarray:
    global _mel_filters
    if _mel_filters is None:
        _mel_filters = librosa.filters.mel(
            sr=TARGET_SR, n_fft=400, n_mels=128,
            fmin=0.0, fmax=8000.0, norm="slaney",
        )
    return _mel_filters


def _log_mel_spectrogram(audio: np.ndarray) -> np.ndarray:
    """Return log-mel spectrogram [1, 128, T] float32 (Whisper params)."""
    # Power spectrogram via STFT with Hann window
    stft = librosa.stft(audio, n_fft=400, hop_length=160,
                        win_length=400, window="hann", center=True)
    magnitudes = np.abs(stft[:, :-1]) ** 2  # drop last frame (parity with Whisper)

    mel = _get_mel_filters() @ magnitudes        # [128, T]
    log_spec = np.log10(np.maximum(mel, 1e-6))  # log10, clamp
    # Per-mel-bin max normalization (Whisper formula)
    log_spec = np.maximum(log_spec, log_spec.max(axis=-1, keepdims=True) - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec[np.newaxis].astype(np.float32)  # [1, 128, T]


# ── Audio-token count formula (mirrors _get_feat_extract_output_lengths) ───────

def _get_audio_token_count(mel_T: int) -> int:
    """Number of encoder output tokens for *mel_T* mel frames."""
    def _conv_out(t: int) -> int:
        return (t + 1) // 2

    leave = mel_T % _CONV_WINDOW
    t = _conv_out(_conv_out(_conv_out(leave)))
    return (mel_T // _CONV_WINDOW) * _TOKENS_PER_WINDOW + t


# ── Prompt construction ────────────────────────────────────────────────────────

def _build_prompt_ids(
    audio_token_count: int,
    user_token: int,
    assistant_token: int,
) -> list[int]:
    """
    Prompt format (from README):
      <|im_start|>system\\n<|im_end|>\\n
      <|im_start|>user\\n<|audio_start|><|audio_pad|>*N<|audio_end|><|im_end|>\\n
      <|im_start|>assistant\\n
    """
    return (
        [IM_START_TOKEN_ID, _SYSTEM_TOKEN, _NEWLINE_TOKEN, IM_END_TOKEN_ID, _NEWLINE_TOKEN]
        + [IM_START_TOKEN_ID, user_token, _NEWLINE_TOKEN]
        + [AUDIO_START_TOKEN_ID]
        + [AUDIO_PAD_TOKEN_ID] * audio_token_count
        + [AUDIO_END_TOKEN_ID, IM_END_TOKEN_ID, _NEWLINE_TOKEN]
        + [IM_START_TOKEN_ID, assistant_token, _NEWLINE_TOKEN]
    )


# ── ORT helpers ────────────────────────────────────────────────────────────────

def _select_providers() -> list[str]:
    available = ort.get_available_providers()
    log.info("Available ORT providers: %s", available)
    ordered = [
        p for p in (
            "MIGraphXExecutionProvider",
            "ROCMExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        )
        if p in available
    ] or ["CPUExecutionProvider"]
    log.info("Using ORT providers: %s", ordered)
    return ordered


def _make_session(path: str, providers: list[str]) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(path, sess_options=opts, providers=providers)
    log.info("Session loaded: %s  providers=%s", Path(path).name, sess.get_providers())
    return sess


# ── Global state ───────────────────────────────────────────────────────────────
_enc_sess:      ort.InferenceSession | None = None
_dec_init_sess: ort.InferenceSession | None = None
_dec_step_sess: ort.InferenceSession | None = None
_tokenizer:     AutoTokenizer | None = None
_embed_tokens:  np.ndarray | None = None   # [vocab, hidden] float16
_user_token:    int = 872                  # resolved at startup
_assistant_token: int = 77091             # resolved at startup


def _model_local_path() -> str:
    return str(Path(MODEL_DIR) / MODEL_NAME.replace("/", "_"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _enc_sess, _dec_init_sess, _dec_step_sess
    global _tokenizer, _embed_tokens, _user_token, _assistant_token

    model_local = _model_local_path()
    providers = _select_providers()

    log.info("Loading ONNX model from %s ...", model_local)
    t0 = time.monotonic()

    # ORT sessions
    _enc_sess      = _make_session(str(Path(model_local) / "encoder.onnx"), providers)
    _dec_init_sess = _make_session(str(Path(model_local) / "decoder_init.onnx"), providers)
    _dec_step_sess = _make_session(str(Path(model_local) / "decoder_step.onnx"), providers)

    # Token embedding matrix — [vocab, hidden] float16
    embed_path = Path(model_local) / "embed_tokens.bin"
    cfg_path   = Path(model_local) / "config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    vocab_size = cfg.get("vocab_size", 151936)
    raw = np.fromfile(str(embed_path), dtype=np.float16)
    hidden = raw.size // vocab_size
    _embed_tokens = raw.reshape(vocab_size, hidden)
    log.info("embed_tokens: shape=%s  dtype=%s", _embed_tokens.shape, _embed_tokens.dtype)

    # Tokenizer — try model dir first (has tokenizer.json), fall back to HF hub
    tok_dir = model_local if (Path(model_local) / "tokenizer.json").exists() else PROCESSOR_NAME
    _tokenizer = AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=True)

    # Resolve "user" and "assistant" sub-word token IDs
    u = _tokenizer.encode("user", add_special_tokens=False)
    a = _tokenizer.encode("assistant", add_special_tokens=False)
    if len(u) == 1:
        _user_token = u[0]
    if len(a) == 1:
        _assistant_token = a[0]
    log.info("user_token=%d  assistant_token=%d", _user_token, _assistant_token)

    # Pre-compute mel filter bank
    _get_mel_filters()

    log.info("Ready in %.1f s", time.monotonic() - t0)
    yield

    _enc_sess = _dec_init_sess = _dec_step_sess = None
    _tokenizer = _embed_tokens = None


# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(title="qwen-asr-onnx", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [{"id": MODEL_NAME, "object": "model"}]}


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form(default=""),
    language: str | None = Form(default=None),
    response_format: str = Form(default="json"),
    temperature: float = Form(default=0.0),
):
    if any(x is None for x in (
        _enc_sess, _dec_init_sess, _dec_step_sess, _tokenizer, _embed_tokens
    )):
        raise HTTPException(status_code=503, detail="Model not loaded")

    # ── Read and decode audio ──────────────────────────────────────────────────
    raw = await file.read()
    try:
        audio_np, sr = sf.read(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot decode audio: {exc}") from exc

    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1)
    audio_np = audio_np.astype(np.float32)
    if sr != TARGET_SR:
        audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=TARGET_SR)
    audio_dur = len(audio_np) / TARGET_SR

    t0 = time.monotonic()

    # ── 1. Log-mel spectrogram ────────────────────────────────────────────────
    mel = _log_mel_spectrogram(audio_np)   # [1, 128, T]
    mel_T = mel.shape[2]

    # ── 2. Encoder ────────────────────────────────────────────────────────────
    audio_features = _enc_sess.run(["audio_features"], {"mel": mel})[0]  # [1, N, 1024]
    audio_token_count = audio_features.shape[1]
    log.info("Encoder: mel_T=%d  audio_tokens=%d", mel_T, audio_token_count)

    # ── 3. Prompt token IDs ───────────────────────────────────────────────────
    prompt_ids = _build_prompt_ids(audio_token_count, _user_token, _assistant_token)
    seq_len = len(prompt_ids)

    # ── 4. Embed prompt tokens ────────────────────────────────────────────────
    ids_arr = np.array(prompt_ids, dtype=np.int64)
    input_embeds = _embed_tokens[ids_arr].astype(np.float32)  # [seq_len, hidden]

    # ── 5. Replace <|audio_pad|> slots with encoder output ───────────────────
    audio_start_idx = prompt_ids.index(AUDIO_PAD_TOKEN_ID)
    audio_end_idx   = audio_start_idx + audio_token_count
    input_embeds[audio_start_idx:audio_end_idx] = audio_features[0].astype(np.float32)

    input_embeds = input_embeds[np.newaxis]          # [1, seq_len, hidden]
    position_ids = np.arange(seq_len, dtype=np.int64)[np.newaxis]  # [1, seq_len]

    # ── 6. Decoder prefill ────────────────────────────────────────────────────
    logits, present_keys, present_values = _dec_init_sess.run(
        ["logits", "present_keys", "present_values"],
        {"input_embeds": input_embeds, "position_ids": position_ids},
    )
    next_token = int(np.argmax(logits[0, -1, :]))
    output_tokens = [next_token]

    # ── 7. Autoregressive decode ───────────────────────────────────────────────
    pos = seq_len
    for _ in range(MAX_NEW_TOKENS - 1):
        if next_token in EOS_TOKEN_IDS:
            break
        token_embed = _embed_tokens[next_token].astype(np.float32)[np.newaxis, np.newaxis, :]
        step_pos = np.array([[pos]], dtype=np.int64)
        logits, present_keys, present_values = _dec_step_sess.run(
            ["logits", "present_keys", "present_values"],
            {
                "input_embeds": token_embed,
                "position_ids": step_pos,
                "past_keys":    present_keys,
                "past_values":  present_values,
            },
        )
        next_token = int(np.argmax(logits[0, -1, :]))
        output_tokens.append(next_token)
        pos += 1

    text = _tokenizer.decode(output_tokens, skip_special_tokens=True).strip()
    elapsed = time.monotonic() - t0
    log.info("Transcribed %.1f s in %.2f s: %r", audio_dur, elapsed, text)

    if response_format == "verbose_json":
        return JSONResponse({"text": text, "duration": audio_dur})
    return JSONResponse({"text": text})


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
