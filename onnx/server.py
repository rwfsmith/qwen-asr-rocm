"""
Qwen3-ASR ONNX inference server — onnxruntime direct (no optimum)
Serves OpenAI-compatible /v1/audio/transcriptions.

Model:     andrewleech/qwen3-asr-0.6b-onnx-int8
Processor: Qwen/Qwen3-ASR-0.6B  (feature extractor + tokenizer)
Backend:   onnxruntime-rocm MIGraphXExecutionProvider (ROCm)
           onnxruntime CPUExecutionProvider (CPU fallback)

The ONNX export follows the Whisper/encoder-decoder convention used by
optimum — we replicate the generate() loop directly so we have no
dependency on optimum (which conflicts with the staging torch version).
"""

from __future__ import annotations

import glob
import io
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
from transformers import AutoProcessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_DIR      = os.environ.get("MODEL_DIR", "/app/models")
MODEL_NAME     = os.environ.get("MODEL_NAME", "andrewleech/qwen3-asr-0.6b-onnx-int8")
PROCESSOR_NAME = os.environ.get("PROCESSOR_NAME", "Qwen/Qwen3-ASR-0.6B")
HOST           = os.environ.get("HOST", "0.0.0.0")
PORT           = int(os.environ.get("PORT", "8001"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "448"))
TARGET_SR      = 16_000   # Qwen3-ASR / Whisper encoder expects 16 kHz


# ── Execution provider helpers ─────────────────────────────────────────────────
def _select_providers() -> list[str]:
    """Return ORT execution providers in priority order."""
    available = ort.get_available_providers()
    log.info("Available ORT providers: %s", available)
    ordered = []
    for preferred in (
        "MIGraphXExecutionProvider",
        "ROCMExecutionProvider",
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ):
        if preferred in available:
            ordered.append(preferred)
    if not ordered:
        ordered = ["CPUExecutionProvider"]
    log.info("Using ORT providers: %s", ordered)
    return ordered


def _find_onnx(model_dir: str, pattern: str) -> str:
    """Glob for a single ONNX file matching *pattern* inside model_dir."""
    matches = glob.glob(str(Path(model_dir) / "**" / pattern), recursive=True)
    if not matches:
        raise FileNotFoundError(f"No ONNX file matching {pattern!r} in {model_dir}")
    if len(matches) > 1:
        log.warning("Multiple matches for %r, using %s", pattern, matches[0])
    return matches[0]


def _make_session(path: str, providers: list[str]) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(path, sess_options=opts, providers=providers)
    log.info("Loaded ORT session: %s  providers=%s", Path(path).name, sess.get_providers())
    return sess


# ── Encoder / Decoder inference helpers ───────────────────────────────────────

def _encode(
    enc_sess: ort.InferenceSession,
    input_features: np.ndarray,
) -> np.ndarray:
    """Run the encoder and return last_hidden_state (B, T, D)."""
    feed = {"input_features": input_features.astype(np.float32)}
    # Some exports also accept attention_mask; skip if not in session inputs
    input_names = {inp.name for inp in enc_sess.get_inputs()}
    if "attention_mask" in input_names:
        feed["attention_mask"] = np.ones(input_features.shape[:2], dtype=np.int64)
    outputs = enc_sess.run(None, feed)
    return outputs[0]  # last_hidden_state


def _decode(
    dec_sess: ort.InferenceSession,
    encoder_hidden_states: np.ndarray,
    processor: AutoProcessor,
    language: str | None,
    max_new_tokens: int,
) -> list[int]:
    """
    Autoregressive decoder loop.

    Handles both:
      • Separate decoder_model.onnx  (no use_cache_branch)
      • Merged decoder_model_merged.onnx  (use_cache_branch: bool)
    """
    dec_inputs = {inp.name for inp in dec_sess.get_inputs()}
    dec_outputs = {out.name for out in dec_sess.get_outputs()}
    has_cache_branch = "use_cache_branch" in dec_inputs

    # Build forced prefix tokens from the processor
    forced_decoder_ids: list[int] = []
    if hasattr(processor, "get_decoder_prompt_ids") and language:
        pairs = processor.get_decoder_prompt_ids(language=language, task="transcribe")
        for _, token_id in pairs:
            forced_decoder_ids.append(token_id)

    # Start token
    bos = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    if bos is None or bos < 0:
        bos = processor.tokenizer.bos_token_id or 0

    generated: list[int] = [bos] + forced_decoder_ids
    past_key_values: dict[str, np.ndarray] = {}

    for step in range(max_new_tokens):
        if step == 0 or not has_cache_branch:
            # First step (or no merging): pass full sequence
            input_ids = np.array([generated], dtype=np.int64)
            use_cache = False
        else:
            # Subsequent steps: pass only the last token
            input_ids = np.array([[generated[-1]]], dtype=np.int64)
            use_cache = True

        feed: dict[str, np.ndarray] = {
            "input_ids": input_ids,
            "encoder_hidden_states": encoder_hidden_states,
        }

        if has_cache_branch:
            feed["use_cache_branch"] = np.array([use_cache], dtype=bool)

        # Inject past key-values from previous iteration
        for k, v in past_key_values.items():
            if k in dec_inputs:
                feed[k] = v

        # Fill any remaining past_key_values with zeros on the first step
        if step == 0:
            for inp in dec_sess.get_inputs():
                if inp.name.startswith("past_key_values") and inp.name not in feed:
                    shape = [
                        1 if (isinstance(d, str) or d == 0) else d
                        for d in inp.shape
                    ]
                    feed[inp.name] = np.zeros(shape, dtype=np.float32)

        results = dec_sess.run(None, feed)

        # Collect present key-values for next iteration
        past_key_values = {}
        for out, val in zip(dec_sess.get_outputs(), results):
            if out.name.startswith("present"):
                # Map present_* → past_key_values.*  (common naming convention)
                kv_name = out.name.replace("present", "past_key_values", 1)
                past_key_values[kv_name] = val

        # logits are always the first output
        logits = results[0]  # (B, T, vocab)
        next_token = int(np.argmax(logits[0, -1, :]))
        generated.append(next_token)

        # Stop at EOS
        eos = processor.tokenizer.eos_token_id
        if eos is not None and next_token == eos:
            break

    return generated


# ── Global state ───────────────────────────────────────────────────────────────
_enc_sess: ort.InferenceSession | None = None
_dec_sess: ort.InferenceSession | None = None
_processor: AutoProcessor | None = None


def _model_local_path() -> str:
    safe = MODEL_NAME.replace("/", "_")
    return str(Path(MODEL_DIR) / safe)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _enc_sess, _dec_sess, _processor

    model_local = _model_local_path()
    providers = _select_providers()

    log.info("Loading ONNX model from %s ...", model_local)
    t0 = time.monotonic()

    enc_path = _find_onnx(model_local, "encoder_model*.onnx")
    # Prefer merged decoder (has KV-cache support in one file), fall back to plain
    try:
        dec_path = _find_onnx(model_local, "decoder_model_merged.onnx")
    except FileNotFoundError:
        dec_path = _find_onnx(model_local, "decoder_model.onnx")

    _enc_sess = _make_session(enc_path, providers)
    _dec_sess = _make_session(dec_path, providers)

    log.info("Loading processor from %s ...", PROCESSOR_NAME)
    _processor = AutoProcessor.from_pretrained(PROCESSOR_NAME)
    log.info("Model + processor ready in %.1f s", time.monotonic() - t0)

    yield  # server is up

    _enc_sess = None
    _dec_sess = None
    _processor = None


# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(title="qwen-asr-onnx", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": MODEL_NAME, "object": "model"}],
    }


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form(default=""),
    language: str | None = Form(default=None),
    response_format: str = Form(default="json"),
    temperature: float = Form(default=0.0),
):
    if _enc_sess is None or _dec_sess is None or _processor is None:
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

    # ── Feature extraction ────────────────────────────────────────────────────
    t0 = time.monotonic()
    inputs = _processor(audio_np, sampling_rate=TARGET_SR, return_tensors="np")
    input_features: np.ndarray = inputs["input_features"]  # (1, 128, T)

    # ── Encoder → Decoder ─────────────────────────────────────────────────────
    encoder_hidden_states = _encode(_enc_sess, input_features)
    token_ids = _decode(
        _dec_sess,
        encoder_hidden_states,
        _processor,
        language=language,
        max_new_tokens=MAX_NEW_TOKENS,
    )

    text = _processor.tokenizer.decode(token_ids, skip_special_tokens=True).strip()
    elapsed = time.monotonic() - t0

    log.info(
        "Transcribed %.1f s of audio in %.2f s: %r",
        len(audio_np) / TARGET_SR,
        elapsed,
        text,
    )

    if response_format == "verbose_json":
        return JSONResponse({"text": text, "duration": len(audio_np) / TARGET_SR})
    return JSONResponse({"text": text})


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
