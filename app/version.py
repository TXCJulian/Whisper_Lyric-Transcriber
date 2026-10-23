import os
import tomllib
from functools import cache
from pathlib import Path

_PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def _pyproject_version() -> str:
    try:
        with _PYPROJECT.open("rb") as f:
            return tomllib.load(f)["project"]["version"]
    except (OSError, KeyError, tomllib.TOMLDecodeError):
        return "unknown"


@cache
def get_version_info() -> dict[str, str]:
    """Return what this build is: version, build number, git commit, image variant.

    CI bakes APP_VERSION/APP_BUILD/APP_COMMIT/IMAGE_VARIANT into the image;
    local runs fall back to pyproject.toml's version and placeholder values.
    """
    return {
        "version": os.getenv("APP_VERSION") or _pyproject_version(),
        "build": os.getenv("APP_BUILD") or "dev",
        "commit": os.getenv("APP_COMMIT") or "unknown",
        "image_variant": os.getenv("IMAGE_VARIANT") or "local",
    }
