#!/bin/sh
set -e

MODEL_DIR="${MODEL_DIR:-/app/models}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-ASR-0.6B}"
MODEL_LOCAL="${MODEL_DIR}/$(echo "${MODEL_NAME}" | tr '/' '_')"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.7}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"

# ── Download model weights if not already cached ──────────────────────────────
if [ ! -d "${MODEL_LOCAL}" ] || [ -z "$(ls -A "${MODEL_LOCAL}" 2>/dev/null)" ]; then
    echo "==> Downloading ${MODEL_NAME} to ${MODEL_LOCAL} ..."
    mkdir -p "${MODEL_LOCAL}"
    huggingface-cli download "${MODEL_NAME}" --local-dir "${MODEL_LOCAL}"
    echo "==> Model download complete"
else
    echo "==> Found: ${MODEL_LOCAL}"
fi

echo "==> Model ready: ${MODEL_LOCAL}"
echo "==> Starting Qwen3-ASR vLLM server on :${PORT} ..."

exec qwen-asr-serve "${MODEL_LOCAL}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-model-len "${MAX_NEW_TOKENS}" \
    --dtype bfloat16
