# Multi-GPU Vendor Support Design

**Date**: 2026-03-08
**Status**: Approved

## Goal

Support NVIDIA (CUDA), Intel Arc (XPU), and AMD Radeon (ROCm) GPUs, with CPU fallback. NVIDIA remains the primary/optimized path. Intel and AMD get GPU-accelerated processing via PyTorch backends.

## Constraints

- faster-whisper (CTranslate2) only supports CUDA and CPU — no XPU or ROCm
- ROCm exposes itself as CUDA via HIP (`torch.version.hip` distinguishes them)
- HDemucs is pure PyTorch, so it works on any PyTorch-supported device
- XPU operator coverage may be incomplete for HDemucs — needs CPU fallback

## GPU Backend Module

New file: `app/gpu_backend.py`

Single source of truth for GPU state. Detects available hardware at startup.

**Detection priority** (auto-detect mode):
1. CUDA (NVIDIA) — `torch.cuda.is_available()` and no `torch.version.hip`
2. XPU (Intel) — `torch.xpu.is_available()`
3. ROCm (AMD) — `torch.cuda.is_available()` with `torch.version.hip`
4. CPU — fallback

**Override**: `GPU_BACKEND` env var (`cuda`, `xpu`, `rocm`, `cpu`) skips auto-detection.

**Exposed interface**:
- `get_backend() -> str` — returns `"cuda"`, `"xpu"`, `"rocm"`, or `"cpu"`
- `get_device() -> torch.device` — PyTorch device for model loading
- `get_device_name() -> str` — human-readable GPU name
- `is_nvidia() -> bool` — shorthand for transcription engine selection

## Transcription Engine Abstraction

New file: `app/transcription_engine.py`

### Engine Selection

| Backend        | Engine          | Reason                              |
|----------------|-----------------|-------------------------------------|
| CUDA (NVIDIA)  | faster-whisper  | Best performance, native CUDA       |
| CPU            | faster-whisper  | CTranslate2 has optimized CPU int8  |
| XPU (Intel)    | OpenAI Whisper  | PyTorch XPU backend via IPEX        |
| ROCm (AMD)     | OpenAI Whisper  | PyTorch ROCm backend                |

### Interface

Base class `TranscriptionEngine` with:
- `transcribe(audio_path, model_name, language, ...) -> segments`
- `load_model(model_name)`
- `unload_model()`

### Implementations

- `FasterWhisperEngine` — wraps current `transcription.py` logic. Used for CUDA and CPU.
- `OpenAIWhisperEngine` — new, uses `openai-whisper` package. Used for XPU and ROCm. Normalizes output to match faster-whisper's segment format so downstream code (correction, LRC/TXT generation) works unchanged.

## Separation Module Changes

Minimal changes to `app/separation.py`.

Replace hardcoded `torch.device("cuda")` with `get_device()` from `gpu_backend`.

- CUDA: works as-is
- ROCm: works as-is (HIP exposes as CUDA device)
- XPU: should work (HDemucs is pure PyTorch). May need `float32` instead of `float16`.
- CPU: already supported as fallback

If XPU operators are unsupported for HDemucs, fall back to CPU for separation only and log a warning.

## Docker Multi-Backend Build

Single `Dockerfile` with `ARG GPU_BACKEND=nvidia`.

### Base Images

| Variant  | Base Image                                     |
|----------|------------------------------------------------|
| `nvidia` | `nvidia/cuda:12.8.0-runtime-ubuntu22.04`       |
| `intel`  | `intel/oneapi-basekit:2025.0.0-devel-ubuntu22.04` |
| `amd`    | `rocm/pytorch:latest`                          |
| `cpu`    | `python:3.11-slim`                             |

### Multi-Stage Structure

```dockerfile
ARG GPU_BACKEND=nvidia

FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04 AS base-nvidia
FROM intel/oneapi-basekit:... AS base-intel
FROM rocm/pytorch:... AS base-amd
FROM python:3.11-slim AS base-cpu

FROM base-${GPU_BACKEND} AS runtime
# Common setup: Python 3.11, ffmpeg, pip install
```

### Requirements

Shared `requirements.txt` for common dependencies. Per-backend files for GPU-specific packages:
- `requirements-nvidia.txt` — torch+cu128
- `requirements-intel.txt` — torch+xpu, intel-extension-for-pytorch
- `requirements-amd.txt` — torch+rocm
- `requirements-cpu.txt` — torch (CPU-only)

### docker-compose.yml

Build args for backend selection. GPU device passthrough varies:
- NVIDIA: `deploy.resources.reservations.devices` (GPU)
- AMD: `devices: ["/dev/kfd", "/dev/dri"]`
- Intel: `devices: ["/dev/dri"]`
- CPU: no device passthrough

### Usage

```bash
docker build --build-arg GPU_BACKEND=intel -t lyric-transcriber:intel .
```

## Health Endpoint

Updated `/health` response:

```json
{
  "status": "healthy",
  "gpu_backend": "xpu",
  "gpu_name": "Intel Arc B580",
  "transcription_engine": "openai-whisper"
}
```

Replaces CUDA-specific fields with backend-agnostic ones.

## New Dependency

`openai-whisper` — always installed for simplicity. Only imported when XPU/ROCm backend is active.

## Files Changed/Added

**New**:
- `app/gpu_backend.py`
- `app/transcription_engine.py`
- `requirements-nvidia.txt`
- `requirements-intel.txt`
- `requirements-amd.txt`
- `requirements-cpu.txt`

**Modified**:
- `app/separation.py`
- `app/pipeline.py`
- `app/main.py`
- `app/models.py`
- `Dockerfile`
- `docker-compose.yml`
- `requirements.txt`
