import os
import logging

import torch

logger = logging.getLogger(__name__)

_backend: str | None = None
_device: torch.device | None = None


def _detect_backend() -> str:
    """Auto-detect GPU backend from available hardware."""
    if torch.cuda.is_available():
        if hasattr(torch.version, "hip") and torch.version.hip:
            logger.info("Detected ROCm (AMD GPU via HIP)")
            return "rocm"
        logger.info("Detected CUDA (NVIDIA GPU)")
        return "cuda"

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        logger.info("Detected XPU (Intel GPU)")
        return "xpu"

    logger.info("No GPU detected, using CPU")
    return "cpu"


def _is_backend_available(backend: str) -> bool:
    """Check if the requested backend is actually usable in this torch build."""
    if backend == "cuda":
        return torch.cuda.is_available()
    if backend == "rocm":
        return torch.cuda.is_available() and hasattr(torch.version, "hip") and bool(torch.version.hip)
    if backend == "xpu":
        return hasattr(torch, "xpu") and torch.xpu.is_available()
    return True  # CPU is always available


def _resolve_backend() -> str:
    """Resolve backend from env var override or auto-detection."""
    override = os.getenv("GPU_BACKEND", "").lower().strip()
    # Accept Docker build-arg names as aliases
    _aliases = {"nvidia": "cuda", "intel": "xpu", "amd": "rocm"}
    override = _aliases.get(override, override)
    valid = ("cuda", "xpu", "rocm", "cpu")
    if override in valid:
        if _is_backend_available(override):
            logger.info(f"GPU backend override: {override}")
            return override
        logger.warning(
            f"GPU_BACKEND='{override}' requested but not available in this "
            "torch build. Falling back to auto-detection."
        )
        return _detect_backend()
    if override:
        logger.warning(
            f"Invalid GPU_BACKEND='{override}', valid options: {valid}. "
            "Falling back to auto-detection."
        )
    return _detect_backend()


def get_backend() -> str:
    """Return the active GPU backend: 'cuda', 'xpu', 'rocm', or 'cpu'."""
    global _backend
    if _backend is None:
        _backend = _resolve_backend()
    return _backend


def get_device() -> torch.device:
    """Return the PyTorch device for the active backend."""
    global _device
    if _device is None:
        backend = get_backend()
        if backend in ("cuda", "rocm"):
            _device = torch.device("cuda")
        elif backend == "xpu":
            _device = torch.device("xpu")
        else:
            _device = torch.device("cpu")
    return _device


def get_device_name() -> str:
    """Return a human-readable GPU name."""
    device = get_device()
    if device.type == "cuda" and torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    if device.type == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.xpu.get_device_name(0)
    return "CPU"


def get_vram_info() -> dict[str, int] | None:
    """Return total VRAM in MB for the active backend, or None (e.g. CPU, no GPU)."""
    device = get_device()
    try:
        if device.type == "cuda" and torch.cuda.is_available():
            _, total = torch.cuda.mem_get_info()
        elif device.type == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
            _, total = torch.xpu.mem_get_info()
        else:
            return None
    except (RuntimeError, AttributeError):
        return None
    return {"total_mb": total // (1024 * 1024)}


# Approximate VRAM requirements per OpenAI's published guidance (PyTorch
# reference implementation, used for openai-whisper on XPU/ROCm):
# https://github.com/openai/whisper#available-models-and-languages
OPENAI_WHISPER_VRAM_MB = {
    "tiny": 1000,
    "base": 1000,
    "small": 2000,
    "medium": 5000,
    "large-v2": 10000,
    "large-v3": 10000,
    "large-v3-turbo": 6000,
}

# faster-whisper (CTranslate2) runs noticeably leaner than the PyTorch
# reference above -- its own published benchmark measures large-v2 at
# 4525MB VRAM (fp16, beam_size=5): https://github.com/SYSTRAN/faster-whisper
# These figures are extrapolated from that single measured data point plus
# each model's parameter count, then rounded up for headroom.
FASTER_WHISPER_VRAM_MB = {
    "tiny": 1000,
    "base": 1000,
    "small": 1500,
    "medium": 3000,
    "large-v2": 5000,
    "large-v3": 5000,
    "large-v3-turbo": 3000,
}


def get_whisper_model_fit(vram: dict[str, int] | None) -> dict[str, bool] | None:
    """Return which Whisper model sizes fit in the given VRAM info, or None if unconstrained (CPU)."""
    if vram is None:
        return None
    total_mb = vram["total_mb"]
    required_mb = FASTER_WHISPER_VRAM_MB if use_faster_whisper() else OPENAI_WHISPER_VRAM_MB
    return {name: total_mb >= required for name, required in required_mb.items()}


def is_nvidia() -> bool:
    """Check if the active backend is NVIDIA CUDA (not ROCm/HIP)."""
    return get_backend() == "cuda"


def use_faster_whisper() -> bool:
    """Check if faster-whisper should be used (CUDA or CPU)."""
    return get_backend() in ("cuda", "cpu")


def empty_cache():
    """Clear GPU memory cache for the active backend."""
    device = get_device()
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device.type == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.empty_cache()
