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


def _resolve_backend() -> str:
    """Resolve backend from env var override or auto-detection."""
    override = os.getenv("GPU_BACKEND", "").lower().strip()
    valid = ("cuda", "xpu", "rocm", "cpu")
    if override in valid:
        logger.info(f"GPU backend override: {override}")
        return override
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
    backend = get_backend()
    if backend in ("cuda", "rocm"):
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    elif backend == "xpu":
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.xpu.get_device_name(0)
    return "CPU"


def is_nvidia() -> bool:
    """Check if the active backend is NVIDIA CUDA (not ROCm/HIP)."""
    return get_backend() == "cuda"


def use_faster_whisper() -> bool:
    """Check if faster-whisper should be used (CUDA or CPU)."""
    return get_backend() in ("cuda", "cpu")


def empty_cache():
    """Clear GPU memory cache for the active backend."""
    backend = get_backend()
    if backend in ("cuda", "rocm") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif backend == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.empty_cache()
