# ─────────────────────────────────────────────────────────────────────────────
# Qwen3-ASR ONNX API – onnxruntime-rocm + MIGraphX backend (AMD GPU / CPU)
#
# Model: andrewleech/qwen3-asr-0.6b-onnx-int8  (int8 quantised ONNX export)
# Serves the same OpenAI-compatible /v1/audio/transcriptions endpoint as
# the vLLM container so the Wyoming proxy works with either.
#
# Build args:
#   GPU_SUPPORT    rocm | cpu        (default: rocm)
#   ROCM_IMAGE     ROCm base image   (default: rocm/dev-ubuntu-24.04:7.2-complete)

ARG GPU_SUPPORT=rocm
ARG ROCM_IMAGE=rocm/dev-ubuntu-24.04:7.2-complete

FROM ubuntu:24.04 AS base-cpu
FROM ${ROCM_IMAGE} AS base-rocm

FROM base-${GPU_SUPPORT}

ARG GPU_SUPPORT=rocm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-venv \
        python3-pip \
        libsndfile1 \
        ffmpeg \
        wget \
        curl \
        ca-certificates \
        git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ── Python virtual environment ─────────────────────────────────────────────────
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# ── ROCm libraries + onnxruntime-migraphx (ROCm build only) ───────────────────
# Mirrors the kokoro-rocm Dockerfile pattern that is known to work:
#   1. apt install migraphx    — system MIGraphX libs (libmigraphx_c.so etc.)
#   2. rocm[libraries,devel]   — gfx1150-tuned HIP/ROCm Python libs
#                                (provides libamdhip64.so.6, libhipblas.so.2…)
#                                from the AMD staging index
#   3. onnxruntime-migraphx    — from PyPI (no staging index needed)
# onnxruntime-migraphx is NOT on the AMD staging index — install from PyPI.
ARG GPU_SUPPORT
RUN if [ "${GPU_SUPPORT}" = "rocm" ]; then \
        echo "==> Installing system MIGraphX libraries via apt..." && \
        apt-get update && apt-get install -y --no-install-recommends migraphx && \
        apt-get clean && rm -rf /var/lib/apt/lists/* && \
        echo "==> Installing gfx1150 ROCm Python libs from AMD staging..." && \
        pip install --no-cache-dir \
            --index-url https://rocm.nightlies.amd.com/v2-staging/gfx1150/ \
            "rocm[libraries,devel]" && \
        echo "==> Installing onnxruntime-migraphx from PyPI..." && \
        pip uninstall -y onnxruntime onnxruntime-rocm 2>/dev/null || true && \
        pip install --no-cache-dir onnxruntime-migraphx; \
    else \
        echo "==> CPU build: installing onnxruntime..." && \
        pip install --no-cache-dir onnxruntime; \
    fi

# ── Python inference stack ─────────────────────────────────────────────────────
# Use onnxruntime directly — optimum conflicts with the ROCm torch version.
# transformers provides AutoProcessor (feature extractor + tokenizer).
# librosa handles sample-rate conversion for non-16 kHz inputs.
RUN pip install --no-cache-dir \
        "transformers>=4.45" \
        "huggingface_hub[cli]>=0.27" \
        "fastapi>=0.115" \
        "uvicorn[standard]>=0.30" \
        "python-multipart>=0.0.9" \
        "soundfile>=0.12" \
        "librosa>=0.10" \
        "numpy>=1.26"

# ── App structure ─────────────────────────────────────────────────────────────
RUN useradd -m -u 1001 appuser \
    && mkdir -p /app/models \
    && chown -R appuser:appuser /app

WORKDIR /app

COPY --chown=appuser:appuser server.py entrypoint.sh ./
RUN sed -i 's/\r$//' /app/entrypoint.sh && chmod +x /app/entrypoint.sh

USER appuser

# ── Runtime environment ───────────────────────────────────────────────────────
ENV MODEL_DIR=/app/models \
    MODEL_NAME=andrewleech/qwen3-asr-0.6b-onnx-int8 \
    PROCESSOR_NAME=Qwen/Qwen3-ASR-0.6B \
    HOST=0.0.0.0 \
    PORT=8001

EXPOSE 8001

ENTRYPOINT ["/app/entrypoint.sh"]
