"""Turn a batch request (a directory or a CSV of paths) into a list of songs.

Every path is checked against TRANSCRIBER_ALLOWED_ROOTS (app/paths.py). A CSV
with any bad line is rejected as a whole, listing every bad line, so a batch is
never started on a partial list.
"""
import csv
import io
from pathlib import Path

from app import config
from app.paths import PathError, resolve_allowed_path


class BatchInputError(Exception):
    """A batch request that can't be turned into a song list (HTTP 422)."""

    status_code = 422

    def __init__(self, code: str, message: str, errors: list[dict] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.errors = errors or []


def _is_audio(path: Path) -> bool:
    return path.suffix.lower() in config.AUDIO_EXTENSIONS


def list_directory_audio(
    directory: str, recursive: bool, roots: list[Path] | None = None
) -> list[Path]:
    """Audio files in *directory* (and subfolders if *recursive*), sorted by name."""
    folder = resolve_allowed_path(directory, roots)
    if not folder.is_dir():
        raise BatchInputError("invalid_batch_request", f"Not a directory: {directory}")

    candidates = folder.rglob("*") if recursive else folder.iterdir()
    songs: list[Path] = []
    for entry in candidates:
        if not _is_audio(entry):
            continue
        # A symlinked file pointing outside the roots is rejected the same way
        # as such a path in a CSV.
        resolved = resolve_allowed_path(str(entry), roots)
        if resolved.is_file():
            songs.append(entry)

    if not songs:
        raise BatchInputError(
            "invalid_batch_request", f"No audio files in directory: {directory}"
        )
    songs.sort(key=lambda p: p.relative_to(folder).as_posix().casefold())
    return [resolve_allowed_path(str(p), roots) for p in songs]


def parse_csv_paths(content: bytes, roots: list[Path] | None = None) -> list[Path]:
    """Song paths from a CSV: one path per row, first column, optional 'path' header."""
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as e:
        raise BatchInputError(
            "invalid_csv",
            "CSV is not valid UTF-8",
            [{"line": 0, "path": "", "reason": str(e)}],
        )

    songs: list[Path] = []
    errors: list[dict] = []
    seen_content = False
    for line_no, row in enumerate(csv.reader(io.StringIO(text, newline="")), start=1):
        cells = [cell.strip() for cell in row]
        if not any(cells) or cells[0].startswith("#"):
            continue
        first_row = not seen_content
        seen_content = True
        if first_row and cells[0].casefold() == "path":
            continue

        raw = cells[0]
        if len([c for c in cells if c]) > 1:
            errors.append({
                "line": line_no,
                "path": ",".join(row),
                "reason": "unexpected extra columns (quote paths containing commas)",
            })
            continue
        if not _is_audio(Path(raw)):
            errors.append({"line": line_no, "path": raw, "reason": "not an audio file"})
            continue
        try:
            resolved = resolve_allowed_path(raw, roots)
        except PathError as e:
            errors.append({"line": line_no, "path": raw, "reason": e.message})
            continue
        if not resolved.is_file():
            errors.append({"line": line_no, "path": raw, "reason": "not a file"})
            continue
        if resolved not in songs:
            songs.append(resolved)

    if errors:
        raise BatchInputError(
            "invalid_csv", f"{len(errors)} invalid line(s) in CSV", errors
        )
    if not songs:
        raise BatchInputError("invalid_batch_request", "CSV contains no song paths")
    return songs


def collect_batch_paths(
    directory: str | None, csv_content: bytes | None, recursive: bool
) -> list[Path]:
    """Song list for a batch request: exactly one of *directory* or *csv_content*."""
    if (directory is None) == (csv_content is None):
        raise BatchInputError(
            "invalid_batch_request", "Provide exactly one of 'directory' or 'csv'"
        )
    if directory is not None:
        songs = list_directory_audio(directory, recursive)
    else:
        songs = parse_csv_paths(csv_content)
    if len(songs) > config.MAX_BATCH_SIZE:
        raise BatchInputError(
            "invalid_batch_request",
            f"Batch has {len(songs)} songs; the limit is {config.MAX_BATCH_SIZE}",
        )
    return songs

