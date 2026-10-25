"""Settings for batch transcription and the Genius correction threshold.

Read once at import from the environment. Kept free of torch imports so it can
be used (and tested) without the ML stack.
"""
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

BATCH_OUTPUT_MODES = ("download", "alongside")

AUDIO_EXTENSIONS = frozenset(
    {".flac", ".mp3", ".m4a", ".ogg", ".opus", ".wav", ".aac", ".wma", ".alac", ".aiff"}
)

MAX_BATCH_SIZE = 5000


def parse_roots(raw: str) -> list[Path]:
    """Parse TRANSCRIBER_ALLOWED_ROOTS: comma-separated directories, resolved, deduplicated."""
    roots: list[Path] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        root = Path(part).resolve()
        if root not in roots:
            roots.append(root)
    return roots


def parse_batch_output(raw: str) -> str:
    """Parse TRANSCRIBER_BATCH_OUTPUT; anything unknown falls back to 'download'."""
    value = raw.strip().lower()
    if value in BATCH_OUTPUT_MODES:
        return value
    logger.error(
        f"Invalid TRANSCRIBER_BATCH_OUTPUT='{raw}', expected one of "
        f"{BATCH_OUTPUT_MODES}; using 'download'"
    )
    return "download"


def parse_min_match(raw: str) -> float:
    """Parse CORRECTION_MIN_MATCH (percent, 0-100); invalid values fall back to 0."""
    if not raw.strip():
        return 0.0
    try:
        value = float(raw)
    except ValueError:
        value = -1.0
    if 0.0 <= value <= 100.0:
        return value
    logger.error(f"Invalid CORRECTION_MIN_MATCH='{raw}', expected 0-100; using 0")
    return 0.0


ALLOWED_ROOTS: list[Path] = parse_roots(os.getenv("TRANSCRIBER_ALLOWED_ROOTS", ""))
BATCH_OUTPUT: str = parse_batch_output(os.getenv("TRANSCRIBER_BATCH_OUTPUT", "download"))
CORRECTION_MIN_MATCH: float = parse_min_match(os.getenv("CORRECTION_MIN_MATCH", ""))


def resolve_min_match(value: float | None) -> float:
    """Per-request threshold if given, else the CORRECTION_MIN_MATCH default."""
    if value is None:
        return CORRECTION_MIN_MATCH
    if not 0.0 <= value <= 100.0:
        raise ValueError(f"correction_min_match must be between 0 and 100, got {value}")
    return float(value)
