"""Path checks for the path-based (batch) endpoints.

Any caller can name a path, so every path is fully resolved (``..`` and
symlinks included) and must land inside one of TRANSCRIBER_ALLOWED_ROOTS --
otherwise e.g. /proc/self/environ (which holds the Genius token) or the model
cache would be readable through the API.
"""
from pathlib import Path

from app import config


class PathError(Exception):
    code = "path_error"
    status_code = 400

    def __init__(self, message: str, path: str):
        super().__init__(message)
        self.message = message
        self.path = path


class PathNotAllowed(PathError):
    code = "path_not_allowed"
    status_code = 403


class PathNotFound(PathError):
    code = "path_not_found"
    status_code = 404


class BatchDisabled(PathError):
    code = "batch_disabled"
    status_code = 403


def resolve_allowed_path(path: str, roots: list[Path] | None = None) -> Path:
    """Resolve *path* and require it to be inside one of the allowed roots."""
    if roots is None:
        roots = config.ALLOWED_ROOTS
    if not roots:
        raise BatchDisabled(
            "Batch processing is disabled: set TRANSCRIBER_ALLOWED_ROOTS", path
        )
    candidate = Path(path)
    if not candidate.is_absolute():
        raise PathNotAllowed("Path must be absolute", path)
    try:
        resolved = candidate.resolve(strict=True)
    except OSError:
        raise PathNotFound("Path does not exist", path)
    # Roots are resolved again here: a root that is (or became) a symlink,
    # e.g. a NAS mount, must still match the resolved file path.
    for root in roots:
        real_root = root.resolve()
        if resolved == real_root or resolved.is_relative_to(real_root):
            return resolved
    raise PathNotAllowed("Path is outside TRANSCRIBER_ALLOWED_ROOTS", path)
