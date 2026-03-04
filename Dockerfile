# ─────────────────────────────────────────────────────────────────────────────
# Qwen3-ASR API – vLLM backend (AMD GPU / CPU)
#
# Build args:
#   GPU_SUPPORT    rocm | cpu        (default: rocm)
#   ROCM_IMAGE     ROCm base image   (default: rocm/dev-ubuntu-24.04:7.2-complete)
#   VLLM_REF       vllm git ref to build from (default: main)
#
# GPU_SUPPORT=rocm install sequence:
#   1. apt install migraphx          — system MIGraphX / HIP libs
#   2. pip install rocm[libraries,devel]  — from AMD gfx1150 staging only
#   3. pip install torch             — from AMD gfx1150 staging only
#   4. pip install triton            — AMD ROCm-compatible triton (PyPI)
#   5. pip install flash-attn        — built with triton AMD backend, no CUDA kernels
#   6. Build vllm from source        — VLLM_TARGET_DEVICE=rocm avoids CUDA wheel
#   7. pip install qwen-asr …        — everything else from PyPI
#
# vLLM is built from source with VLLM_TARGET_DEVICE=rocm so it compiles its
# ROCm custom ops (paged attention, etc.) rather than pulling CUDA wheels.
# flash-attn is built with FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE.

ARG GPU_SUPPORT=rocm
ARG ROCM_IMAGE=rocm/dev-ubuntu-24.04:7.2-complete
ARG VLLM_REF=main
ARG ROCM_ARCH=gfx1150

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
        cmake \
        ninja-build \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ── Python virtual environment ─────────────────────────────────────────────────
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# ── ROCm system libraries (ROCm build only) ───────────────────────────────────
ARG GPU_SUPPORT
RUN if [ "${GPU_SUPPORT}" = "rocm" ]; then \
        echo "==> Installing system MIGraphX libraries via apt..." && \
        apt-get update && apt-get install -y --no-install-recommends migraphx && \
        apt-get clean && rm -rf /var/lib/apt/lists/* && \
        echo "==> Installing amdsmi Python bindings from system ROCm..." && \
        pip install --no-cache-dir /opt/rocm/share/amd_smi/; \
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
            torch torchaudio; \
    else \
        echo "==> CPU build: installing torch/torchaudio from pytorch.org..." && \
        pip install --no-cache-dir torch torchaudio \
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

# ── vLLM: source build for ROCm, PyPI wheel for CPU ──────────────────────────
# The PyPI vllm wheel only ships CUDA kernels.  For ROCm we must build from
# source with VLLM_TARGET_DEVICE=rocm so it compiles HIP/ROCm paged-attention
# ops and does not pull any NVIDIA CUDA packages.
# VLLM_REF selects the git ref (tag, branch, or commit) to build.
ARG GPU_SUPPORT
ARG VLLM_REF
ARG ROCM_ARCH
RUN if [ "${GPU_SUPPORT}" = "rocm" ]; then \
        TORCH_VER=$(pip show torch | grep ^Version | cut -d' ' -f2) && \
        echo "==> Cloning vllm @ ${VLLM_REF} for ROCm source build..." && \
        git clone --depth 1 --branch ${VLLM_REF} \
            https://github.com/vllm-project/vllm /tmp/vllm && \
        cd /tmp/vllm && \
        echo "==> Installing vllm ROCm build requirements (pinning torch==${TORCH_VER})..." && \
        ROCM_REQ=$([ -f requirements/rocm.txt ] && echo requirements/rocm.txt || echo requirements-rocm.txt) && \
        pip install --no-cache-dir -r "${ROCM_REQ}" \
            "torch==${TORCH_VER}" \
            --extra-index-url https://rocm.nightlies.amd.com/v2-staging/gfx1150/ && \
        echo "==> Building vllm with VLLM_TARGET_DEVICE=rocm (arch=${ROCM_ARCH})..." && \
        rocm-sdk init && \
        ROCM_PATH=$(rocm-sdk path --root) && \
        echo "==> Detected ROCM_PATH: ${ROCM_PATH}" && \
        VLLM_TARGET_DEVICE=rocm \
        MAX_JOBS=$(nproc) \
        SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0 \
        ROCM_PATH=${ROCM_PATH} \
        PYTORCH_ROCM_ARCH=${ROCM_ARCH} \
            pip install --no-cache-dir --no-build-isolation . && \
        rm -rf /tmp/vllm; \
    else \
        echo "==> CPU build: installing vllm from PyPI..." && \
        pip install --no-cache-dir vllm; \
    fi

# ── qwen-asr and runtime deps ─────────────────────────────────────────────────
RUN pip install --no-cache-dir \
        "qwen-asr>=0.0.6" \
        "huggingface_hub[cli]>=0.27" \
        "fastapi>=0.115" \
        "uvicorn[standard]>=0.30"

# ── Remove Ray (pulled in by vllm) ────────────────────────────────────────────
# Ray is only needed for multi-node / multi-GPU distributed inference.
# On single-GPU deployments it is unused, and its bundled pyamdsmi triggers a
# Bus error (SIGBUS) on gfx1150 when it tries to enumerate AMD GPUs via
# ctypes.LoadLibrary during import.  vLLM gracefully skips Ray when absent.
RUN pip uninstall -y ray 2>/dev/null || true

# ── Bake ROCm pip-package paths so torch can find HIP libs at runtime ─────────
# rocm-sdk init sets up the SDK; we capture ROCM_PATH and write it to
# /opt/venv/rocm-env.sh (sourced by entrypoint.sh) and to ld.so.conf.d so
# the dynamic linker can resolve libamdhip64.so etc without an explicit
# LD_LIBRARY_PATH export from the user.
ARG GPU_SUPPORT
RUN if [ "${GPU_SUPPORT}" = "rocm" ]; then \
        rocm-sdk init && \
        ROCM_PATH=$(rocm-sdk path --root) && \
        echo "==> Baking ROCM_PATH=${ROCM_PATH} into image..." && \
        echo "${ROCM_PATH}/lib" > /etc/ld.so.conf.d/rocm-pip.conf && \
        ldconfig && \
        printf 'export ROCM_PATH="%s"\nexport LD_LIBRARY_PATH="%s/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"\nexport PATH="%s/bin:${PATH}"\n' \
            "${ROCM_PATH}" "${ROCM_PATH}" "${ROCM_PATH}" > /opt/venv/rocm-env.sh && \
        echo "==> /opt/venv/rocm-env.sh written:"; cat /opt/venv/rocm-env.sh; \
    fi

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
    VLLM_TARGET_DEVICE=rocm

EXPOSE 8000

ENTRYPOINT ["/app/entrypoint.sh"]
