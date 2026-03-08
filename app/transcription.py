import os
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


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
