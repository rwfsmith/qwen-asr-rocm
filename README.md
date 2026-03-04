# qwen-asr-rocm

Dockerised **Qwen3-ASR-0.6B** speech-to-text service with a [Wyoming protocol](https://github.com/rhasspy/wyoming) sidecar for direct integration with [Home Assistant](https://www.home-assistant.io/).

Built on AMD ROCm with a **vLLM** backend and **Flash Attention 2** via the AMD triton backend — no NVIDIA/CUDA required.

---

## Features

- **Qwen3-ASR-0.6B** — 52-language ASR, 2000× throughput at batch 128
- **vLLM** inference backend with OpenAI-compatible `/v1/audio/transcriptions` API
- **Flash Attention 2** compiled with AMD triton kernels (`FLASH_ATTENTION_TRITON_AMD_ENABLE`)
- **Wyoming STT sidecar** on port 10300 — plug straight into Home Assistant
- gfx1150 (Ryzen AI / Strix Point) native PyTorch from AMD staging index
- Automatic model download on first start; mount a volume to cache across restarts

---

## Requirements

- AMD GPU with ROCm 7.2 support (tested on Ryzen AI 9 HX 370 / gfx1150)
- Docker with `/dev/kfd` and `/dev/dri` accessible
- ~2 GB GPU VRAM (0.6B model), ~3 GB disk for model weights

> **TrueNAS SCALE users:** the kernel already includes `amdgpu` / `amdkfd` — no host ROCm installation needed.

---

## Quick start

### TrueNAS SCALE / Linux

```bash
git clone https://github.com/rwfsmith/qwen-asr-rocm.git
cd qwen-asr-rocm

export VIDEO_GID=$(getent group video | cut -d: -f3)
export RENDER_GID=$(getent group render | cut -d: -f3)
export QWEN_ASR_DATA_DIR=/mnt/tank/apps/qwen-asr/data   # adjust to your pool

sudo -E mkdir -p "$QWEN_ASR_DATA_DIR"/{models,miopen-cache,hf-cache}
docker compose -f truenas-compose.yml up -d --build
```

The container will:
1. Build the image (~20–40 min first time — flash-attn compiles from source)
2. Download `Qwen/Qwen3-ASR-0.6B` (~1.2 GB) on first start
3. Start the vLLM API on **port 8000** and the Wyoming proxy on **port 10300**

### Development / other Linux

```bash
GPU_SUPPORT=rocm docker compose up -d --build
```

Or CPU-only (slower, transformers backend):

```bash
GPU_SUPPORT=cpu docker compose up -d --build
```

---

## Build arguments

| Arg | Default | Description |
|-----|---------|-------------|
| `GPU_SUPPORT` | `rocm` | `rocm` or `cpu` |
| `ROCM_IMAGE` | `rocm/dev-ubuntu-24.04:7.2-complete` | ROCm base image |

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `Qwen/Qwen3-ASR-0.6B` | HuggingFace model ID |
| `GPU_MEMORY_UTILIZATION` | `0.7` | Fraction of VRAM for vLLM KV cache |
| `MAX_NEW_TOKENS` | `1024` | Max output tokens (increase for long audio) |
| `HOST` | `0.0.0.0` | vLLM listen address |
| `PORT` | `8000` | vLLM listen port |
| `MIOPEN_FIND_MODE` | `3` | MIOpen kernel search mode (exhaustive on first run) |

---

## Install sequence (GPU build)

```
ROCm base image (rocm/dev-ubuntu-24.04:7.2-complete)
  └─ apt install migraphx           ← system MIGraphX C libs
  └─ pip install rocm[libraries,devel]   ── AMD gfx1150 staging index only
  └─ pip install torch                   ── AMD gfx1150 staging index only
  └─ pip install triton                  ── PyPI (AMD ROCm-compatible)
  └─ pip install flash-attn              ── PyPI, built with FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
  └─ pip install vllm qwen-asr …         ── PyPI
```

PyTorch and ROCm libs come exclusively from the AMD gfx1150 staging index to guarantee gfx1150-native kernels. Everything else comes from PyPI so ROCm-specific index overrides don't accidentally pull CUDA builds of vLLM.

---

## API

The vLLM server exposes a standard OpenAI-compatible API:

### Transcribe audio file

```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=Qwen/Qwen3-ASR-0.6B
```

### Check available models

```bash
curl http://localhost:8000/v1/models
```

### Python example

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

with open("audio.wav", "rb") as f:
    result = client.audio.transcriptions.create(
        model="Qwen/Qwen3-ASR-0.6B",
        file=f,
    )
print(result.text)
```

---

## Home Assistant integration

The Wyoming sidecar exposes port **10300**.

1. In Home Assistant: **Settings → Devices & Services → Add Integration → Wyoming Protocol**
2. Host: `<your-truenas-ip>`, Port: `10300`
3. Set as the STT engine in your voice assistant pipeline

> **Note:** The first transcription request may take 30–60 s while MIOpen compiles kernel caches. Subsequent requests are fast.

---

## Volumes

| Mount | Purpose |
|-------|---------|
| `/app/models` | Downloaded model weights (persist across restarts) |
| `/root/.cache/miopen` | MIOpen compiled kernel cache |
| `/root/.cache/huggingface` | HuggingFace hub download cache |

---

## Model options

To use the 1.7B model instead (better accuracy, ~2.5× more VRAM):

```yaml
environment:
  - MODEL_NAME=Qwen/Qwen3-ASR-1.7B
  - GPU_MEMORY_UTILIZATION=0.85
```

---

## License

Model weights: [Qwen License](https://huggingface.co/Qwen/Qwen3-ASR-0.6B/blob/main/LICENSE)  
This repo: MIT
