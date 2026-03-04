# ─────────────────────────────────────────────────────────────────────────────
# Qwen3-ASR API – vLLM backend (AMD GPU / CPU)
#
# Build args:
#   GPU_SUPPORT    rocm | cpu        (default: rocm)
#   ROCM_IMAGE     ROCm base image   (default: rocm/dev-ubuntu-24.04:7.2-complete)
#
# GPU_SUPPORT=rocm install sequence:
#   1. apt install migraphx          — system MIGraphX / HIP libs
#   2. pip install rocm[libraries,devel]  — from AMD gfx1150 staging only
#   3. pip install torch             — from AMD gfx1150 staging only
#   4. pip install triton            — AMD ROCm-compatible triton (PyPI)
#   5. pip install flash-attn        — built with triton AMD backend, no CUDA kernels
#   6. pip install vllm qwen-asr …   — everything else from PyPI
#
# flash-attn is built with FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE so it uses
# triton kernels instead of CUDA device code, and is passed through to vLLM /
# qwen-asr via VLLM_USE_TRITON_FLASH_ATTN at runtime.
#
# GPU_SUPPORT=cpu: CPU torch from pytorch.org, transformers backend.
#
# Model (Qwen/Qwen3-ASR-0.6B) is downloaded by entrypoint.sh on first start.
# ─────────────────────────────────────────────────────────────────────────────

ARG GPU_SUPPORT=rocm
ARG ROCM_IMAGE=rocm/dev-ubuntu-24.04:7.2-complete

# Named base stages — Docker selects via FROM base-${GPU_SUPPORT}
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
        build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ── Python virtual environment ─────────────────────────────────────────────────
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# ── ROCm system libraries (ROCm build only) ───────────────────────────────────
ARG GPU_SUPPORT
RUN if [ "${GPU_SUPPORT}" = "rocm" ]; then \
        echo "==> Installing system MIGraphX libraries via apt..." && \
        apt-get update && apt-get install -y --no-install-recommends migraphx && \
        apt-get clean && rm -rf /var/lib/apt/lists/*; \
    fi

# ── PyTorch: staging index for ROCm, pytorch.org for CPU ──────────────────────
# Separate install so --index-url applies only to these packages.
ARG GPU_SUPPORT
RUN if [ "${GPU_SUPPORT}" = "rocm" ]; then \
        echo "==> Installing rocm[libraries,devel] from AMD gfx1150 staging..." && \
        pip install --no-cache-dir \
            --index-url https://rocm.nightlies.amd.com/v2-staging/gfx1150/ \
            "rocm[libraries,devel]" && \
        echo "==> Installing torch from AMD gfx1150 staging..." && \
        pip install --no-cache-dir \
            --index-url https://rocm.nightlies.amd.com/v2-staging/gfx1150/ \
            torch; \
    else \
        echo "==> CPU build: installing torch from pytorch.org..." && \
        pip install --no-cache-dir torch \
            --index-url https://download.pytorch.org/whl/cpu; \
    fi

# ── Triton + Flash Attention 2 (ROCm build only) ──────────────────────────────
# triton is AMD ROCm-compatible and required for flash-attn's triton backend.
# FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE builds flash-attn with triton kernels
# instead of CUDA device code — the only option that works on ROCm.
ARG GPU_SUPPORT
RUN if [ "${GPU_SUPPORT}" = "rocm" ]; then \
        echo "==> Installing triton..." && \
        pip install --no-cache-dir triton && \
        echo "==> Building flash-attn with AMD triton backend..." && \
        FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
            pip install --no-cache-dir flash-attn --no-build-isolation; \
    fi

# ── vLLM + qwen-asr: everything from PyPI ─────────────────────────────────────
# vllm is installed from PyPI here. The ROCm torch already on PATH ensures the
# PyPI vllm picks up the correct GPU libs at runtime rather than re-downloading
# CUDA torch. qwen-asr is installed without [vllm] extra since vllm is already
# present, avoiding any CUDA vllm being pulled as a dep.
RUN pip install --no-cache-dir \
        vllm \
        "qwen-asr>=0.0.6" \
        "huggingface_hub[cli]>=0.27" \
        "fastapi>=0.115" \
        "uvicorn[standard]>=0.30"

# ── App structure ─────────────────────────────────────────────────────────────
RUN useradd -m -u 1001 appuser \
    && mkdir -p /app/models \
    && chown -R appuser:appuser /app

WORKDIR /app

COPY --chown=appuser:appuser entrypoint.sh ./
RUN sed -i 's/\r$//' /app/entrypoint.sh && chmod +x /app/entrypoint.sh

USER appuser

# ── Runtime environment ───────────────────────────────────────────────────────
ENV MODEL_DIR=/app/models \
    MODEL_NAME=Qwen/Qwen3-ASR-0.6B \
    GPU_MEMORY_UTILIZATION=0.7 \
    MAX_NEW_TOKENS=1024 \
    HOST=0.0.0.0 \
    PORT=8000 \
    # Enable flash-attn AMD triton backend at runtime
    FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
    VLLM_USE_TRITON_FLASH_ATTN=1

EXPOSE 8000

ENTRYPOINT ["/app/entrypoint.sh"]
