import os
import logging

import numpy as np
import torch
import torchaudio
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS
import soundfile as sf

from app.gpu_backend import get_device, empty_cache

logger = logging.getLogger(__name__)

_model = None
_device = None
_sample_rate = None


def load_model():
    global _model, _device, _sample_rate
    if _model is None:
        _device = get_device()
        bundle = HDEMUCS_HIGH_MUSDB_PLUS
        _sample_rate = bundle.sample_rate
        _model = bundle.get_model().to(_device)
        _model.eval()
        logger.info(f"HDemucs model loaded on {_device}")
    return _model, _device, _sample_rate


def unload_model():
    """Move HDemucs model off GPU to free VRAM for other models."""
    global _model, _device
    if _model is not None:
        _model.cpu()
        del _model
        _model = None
        empty_cache()
        logger.info("HDemucs model unloaded from GPU")


def separate_vocals(input_path: str, output_dir: str) -> str:
    """Separate vocals from audio. Returns path to vocals WAV file."""
    model, device, target_sr = load_model()
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

    # Process in chunks to avoid GPU OOM on long tracks
    chunk_seconds = 30
    overlap_seconds = 1
    chunk_size = chunk_seconds * target_sr
    overlap_size = overlap_seconds * target_sr
    total_length = waveform.shape[1]

    if total_length <= chunk_size:
        # Short enough to process in one go
        with torch.no_grad():
            sources = model(waveform.unsqueeze(0).to(device))
        vocals = sources[0, 3].cpu()
    else:
        # Chunked processing with overlap
        logger.info(
            f"Audio is {total_length / target_sr:.0f}s, processing in {chunk_seconds}s chunks"
        )
        vocals = torch.zeros_like(waveform)
        weight = torch.zeros(1, total_length)
        pos = 0
        while pos < total_length:
            end = min(pos + chunk_size, total_length)
            chunk = waveform[:, pos:end]

            # Pad last chunk if too short
            if chunk.shape[1] < chunk_size:
                pad = torch.zeros(2, chunk_size - chunk.shape[1])
                chunk = torch.cat([chunk, pad], dim=1)

            with torch.no_grad():
                sources = model(chunk.unsqueeze(0).to(device))
            chunk_vocals = sources[0, 3].cpu()

            actual_len = min(chunk_size, total_length - pos)
            vocals[:, pos : pos + actual_len] += chunk_vocals[:, :actual_len]
            weight[:, pos : pos + actual_len] += 1.0

            pos += chunk_size - overlap_size

        # Average overlapping regions
        weight = weight.clamp(min=1.0)
        vocals = vocals / weight

    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(input_path))[0]
    vocals_path = os.path.join(output_dir, f"{stem}_vocals.wav")
    sf.write(vocals_path, vocals.numpy().T, target_sr)

    logger.info(f"Vocals separated: {vocals_path}")
    return vocals_path
