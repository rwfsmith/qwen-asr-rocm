#!/bin/sh
set -e

MODEL_DIR="${MODEL_DIR:-/app/models}"
MODEL_NAME="${MODEL_NAME:-andrewleech/qwen3-asr-0.6b-onnx-int8}"
PROCESSOR_NAME="${PROCESSOR_NAME:-Qwen/Qwen3-ASR-0.6B}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8001}"
MODEL_LOCAL="${MODEL_DIR}/$(echo "${MODEL_NAME}" | tr '/' '_')"

# ── Download ONNX model weights if not already cached ─────────────────────────
if [ ! -d "${MODEL_LOCAL}" ] || [ -z "$(ls -A "${MODEL_LOCAL}" 2>/dev/null)" ]; then
    echo "==> Downloading ${MODEL_NAME} to ${MODEL_LOCAL} ..."
    mkdir -p "${MODEL_LOCAL}"
    huggingface-cli download "${MODEL_NAME}" --local-dir "${MODEL_LOCAL}"
    echo "==> ONNX model download complete"
else
    echo "==> Found: ${MODEL_LOCAL}"
fi

# ── Download processor from base model if not already cached ──────────────────
PROCESSOR_LOCAL="${MODEL_DIR}/$(echo "${PROCESSOR_NAME}" | tr '/' '_')"
if [ ! -d "${PROCESSOR_LOCAL}" ] || [ -z "$(ls -A "${PROCESSOR_LOCAL}" 2>/dev/null)" ]; then
    echo "==> Downloading processor ${PROCESSOR_NAME} to ${PROCESSOR_LOCAL} ..."
    mkdir -p "${PROCESSOR_LOCAL}"
    huggingface-cli download "${PROCESSOR_NAME}" \
        --local-dir "${PROCESSOR_LOCAL}" \
        --include "*.json" "*.tiktoken" "*.model" "preprocessor_config.json"
    echo "==> Processor download complete"
else
    echo "==> Found processor: ${PROCESSOR_LOCAL}"
fi

echo "==> Starting Qwen3-ASR ONNX server on :${PORT} ..."
exec python3 /app/server.py
