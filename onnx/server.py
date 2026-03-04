"""
Qwen3-ASR ONNX inference server
Serves OpenAI-compatible /v1/audio/transcriptions so the Wyoming proxy
and any existing tooling works identically to the vLLM backend.

Model:     andrewleech/qwen3-asr-0.6b-onnx-int8
Processor: Qwen/Qwen3-ASR-0.6B  (provides feature extractor + tokenizer)
Backend:   onnxruntime-rocm with MIGraphXExecutionProvider (ROCm)
           onnxruntime with CPUExecutionProvider (CPU fallback)
"""

from __future__ import annotations

import io
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
from transformers import AutoProcessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_DIR     = os.environ.get("MODEL_DIR", "/app/models")
MODEL_NAME    = os.environ.get("MODEL_NAME", "andrewleech/qwen3-asr-0.6b-onnx-int8")
PROCESSOR_NAME = os.environ.get("PROCESSOR_NAME", "Qwen/Qwen3-ASR-0.6B")
HOST          = os.environ.get("HOST", "0.0.0.0")
PORT          = int(os.environ.get("PORT", "8001"))
TARGET_SR     = 16_000   # Qwen3-ASR / Whisper encoder expects 16 kHz

# ── Execution provider selection ───────────────────────────────────────────────
def _select_provider() -> str:
    """Pick the best available ORT execution provider."""
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
        log.info("Available ORT providers: %s", available)
        for preferred in ("MIGraphXExecutionProvider", "ROCMExecutionProvider",
                          "CUDAExecutionProvider", "CPUExecutionProvider"):
            if preferred in available:
                log.info("Using ORT provider: %s", preferred)
                return preferred
    except Exception as exc:
        log.warning("Could not query ORT providers: %s", exc)
    return "CPUExecutionProvider"


# ── Global model state ─────────────────────────────────────────────────────────
_model: ORTModelForSpeechSeq2Seq | None = None
_processor: AutoProcessor | None = None
_provider: str = "CPUExecutionProvider"
_model_local: str = ""


def _model_local_path() -> str:
    """Derive the local model directory path from MODEL_NAME."""
    safe = MODEL_NAME.replace("/", "_")
    return str(Path(MODEL_DIR) / safe)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _processor, _provider, _model_local

    _model_local = _model_local_path()
    _provider = _select_provider()

    log.info("Loading ONNX model from %s ...", _model_local)
    t0 = time.monotonic()
    _model = ORTModelForSpeechSeq2Seq.from_pretrained(
        _model_local,
        provider=_provider,
        use_io_binding=False,   # IO binding not always supported on MIGraphX
    )
    log.info("Loading processor from %s ...", PROCESSOR_NAME)
    _processor = AutoProcessor.from_pretrained(PROCESSOR_NAME)
    log.info("Model + processor ready in %.1f s", time.monotonic() - t0)

    yield   # server is up

    _model = None
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
    if _model is None or _processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # ── Read and decode audio ──────────────────────────────────────────────────
    raw = await file.read()
    try:
        audio_np, sr = sf.read(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot decode audio: {exc}") from exc

    # Convert to mono float32
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1)
    audio_np = audio_np.astype(np.float32)

    # Resample to 16 kHz if needed
    if sr != TARGET_SR:
        audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=TARGET_SR)

    # ── Feature extraction + inference ────────────────────────────────────────
    t0 = time.monotonic()
    inputs = _processor(
        audio_np,
        sampling_rate=TARGET_SR,
        return_tensors="pt",
    )

    generate_kwargs: dict = {"max_new_tokens": 448}
    if language:
        generate_kwargs["language"] = language

    predicted_ids = _model.generate(**inputs, **generate_kwargs)
    text: str = _processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
    elapsed = time.monotonic() - t0

    log.info("Transcribed %.1f s of audio in %.2f s: %r", len(audio_np) / TARGET_SR, elapsed, text)

    if response_format in ("verbose_json",):
        return JSONResponse({"text": text, "duration": len(audio_np) / TARGET_SR})
    return JSONResponse({"text": text})


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
