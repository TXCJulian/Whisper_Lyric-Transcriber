import os
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

_models: dict[str, WhisperModel] = {}


def _is_cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def get_model(model_size: str = "large-v3-turbo") -> WhisperModel:
    """Load and cache a faster-whisper model."""
    if model_size not in _models:
        device = "cuda" if _is_cuda_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        logger.info(f"Loading Whisper model '{model_size}' on {device} ({compute_type})")
        _models[model_size] = WhisperModel(
            model_size, device=device, compute_type=compute_type
        )
    return _models[model_size]


@dataclass
class WordTiming:
    start: float
    end: float
    word: str


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list[WordTiming] = field(default_factory=list)


def transcribe(
    audio_path: str,
    model_size: str = "large-v3-turbo",
    language: str | None = None,
    artist: str | None = None,
    title: str | None = None,
    language_callback: Callable[[str], None] | None = None,
) -> tuple[list[Segment], str]:
    """Transcribe audio file. Returns (segments, detected_language)."""
    model = get_model(model_size)

    kwargs: dict[str, Any] = {
        "word_timestamps": True,
        "vad_filter": True,
        "condition_on_previous_text": False,
    }
    if language:
        kwargs["language"] = language

    # Prime Whisper with artist/title to improve recognition of names and style
    if artist and title:
        kwargs["initial_prompt"] = f"{artist} - {title}"
        logger.info(f"Using initial_prompt: '{kwargs['initial_prompt']}'")
    elif artist:
        kwargs["initial_prompt"] = artist
    elif title:
        kwargs["initial_prompt"] = title

    segments_iter, info = model.transcribe(audio_path, **kwargs)
    detected_language = info.language
    logger.info(f"Detected language: {detected_language} (prob: {info.language_probability:.2f})")

    if language_callback:
        language_callback(detected_language)

    results = []
    for seg in segments_iter:
        words = []
        if seg.words:
            words = [WordTiming(start=w.start, end=w.end, word=w.word) for w in seg.words]
        results.append(
            Segment(start=seg.start, end=seg.end, text=seg.text.strip(), words=words)
        )

    logger.info(f"Transcription complete: {len(results)} segments")
    return results, detected_language


def segments_to_lrc(segments: list[Segment]) -> str:
    """Convert segments to LRC format."""
    lines = []
    for seg in segments:
        minutes = int(seg.start // 60)
        seconds = seg.start % 60
        lines.append(f"[{minutes:02d}:{seconds:05.2f}] {seg.text}")
    return "\n".join(lines)


def segments_to_txt(segments: list[Segment]) -> str:
    """Convert segments to plain text."""
    return "\n".join(seg.text for seg in segments)


def write_output_files(
    segments: list[Segment],
    output_dir: str,
    stem: str,
    format: str = "lrc",
) -> list[str]:
    """Write transcription segments to output files. Returns list of created file paths."""
    os.makedirs(output_dir, exist_ok=True)
    output_files: list[str] = []

    if format in ("lrc", "all"):
        lrc_path = os.path.join(output_dir, f"{stem}.lrc")
        with open(lrc_path, "w", encoding="utf-8") as f:
            f.write(segments_to_lrc(segments))
        output_files.append(lrc_path)

    if format in ("txt", "all"):
        txt_path = os.path.join(output_dir, f"{stem}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(segments_to_txt(segments))
        output_files.append(txt_path)

    return output_files
