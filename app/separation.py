import os
import logging

import numpy as np
import torch
import torchaudio
import soundfile as sf
from demucs.pretrained import get_model
from demucs.apply import apply_model

from app.gpu_backend import get_device, get_backend, empty_cache

logger = logging.getLogger(__name__)

DEFAULT_DEMUCS_MODEL = os.getenv("DEMUCS_MODEL", "htdemucs")
_segment_env = os.getenv("DEMUCS_SEGMENT_SECONDS")
DEMUCS_SEGMENT_SECONDS = float(_segment_env) if _segment_env else None

_model = None
_model_name = None
_device = None
_sample_rate = None
_vocals_index = None


def _load(model_name: str):
    global _model, _model_name, _device, _sample_rate, _vocals_index
    _device = get_device()
    try:
        model = get_model(name=model_name).to(_device)
    except (RuntimeError, NotImplementedError, AssertionError):
        if get_backend() == "xpu":
            logger.warning(
                "Demucs failed to load on XPU (unsupported operators), "
                "falling back to CPU for vocal separation"
            )
            _device = torch.device("cpu")
            model = get_model(name=model_name).to(_device)
        else:
            raise
    model.eval()
    _model = model
    _model_name = model_name
    _sample_rate = model.samplerate
    _vocals_index = model.sources.index("vocals")
    logger.info(f"Demucs model '{model_name}' loaded on {_device}")


def load_model(model_name: str = DEFAULT_DEMUCS_MODEL):
    if _model is None or _model_name != model_name:
        _load(model_name)
    return _model, _device, _sample_rate


def unload_model():
    """Move Demucs model off GPU to free VRAM for other models."""
    global _model, _model_name, _device
    if _model is not None:
        _model.cpu()
        del _model
        _model = None
        _model_name = None
        empty_cache()
        logger.info("Demucs model unloaded from GPU")


def separate_vocals(
    input_path: str, output_dir: str, model_name: str = DEFAULT_DEMUCS_MODEL
) -> str:
    """Separate vocals from audio. Returns path to vocals WAV file."""
    model, device, target_sr = load_model(model_name)
    assert target_sr is not None, "Model sample rate not initialized"

    # Load with soundfile directly to avoid torchaudio's torchcodec dependency
    data, sr = sf.read(input_path, dtype="float32")  # shape: (samples,) or (samples, channels)
    if data.ndim == 1:
        data = data[:, np.newaxis]  # mono -> (samples, 1)
    waveform = torch.from_numpy(data.T)  # -> (channels, samples)

    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)

    # Ensure stereo
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    elif waveform.shape[0] > 2:
        waveform = waveform[:2]

    # Normalize before inference and restore levels after, matching demucs'
    # own separation script (models are trained on normalized audio).
    ref = waveform.mean(0)
    mean, std = ref.mean(), ref.std()
    normalized = (waveform - mean) / std

    # apply_model splits long tracks into overlapping segments internally
    # (bounded by the model's trained segment length) to stay within VRAM.
    with torch.no_grad():
        sources = apply_model(
            model,
            normalized.unsqueeze(0).to(device),
            device=device,
            split=True,
            overlap=0.25,
            segment=DEMUCS_SEGMENT_SECONDS,
            progress=False,
        )[0]

    vocals = (sources[_vocals_index] * std + mean).cpu()

    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(input_path))[0]
    vocals_path = os.path.join(output_dir, f"{stem}_vocals.wav")
    sf.write(vocals_path, vocals.numpy().T, target_sr)

    logger.info(f"Vocals separated: {vocals_path}")
    return vocals_path
