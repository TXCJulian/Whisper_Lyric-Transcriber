"""OpenAPI documentation shared by the routes: API description, tags, form
fields and response definitions. Form fields are built by functions so every
endpoint gets its own FieldInfo instance with the same wording."""
from fastapi import File, Form

from app.gpu_backend import OPENAI_WHISPER_VRAM_MB
from app.models import ErrorResponse, OutputFormat
from app.separation import DEFAULT_DEMUCS_MODEL

WHISPER_MODELS = ", ".join(f"`{name}`" for name in OPENAI_WHISPER_VRAM_MB)

API_DESCRIPTION = """
Automated song lyrics transcription: **Demucs** vocal separation → **Whisper**
speech-to-text → optional **Genius** lyrics correction, producing `.lrc` /
`.txt` files.

## How jobs work

Every processing endpoint answers immediately with a `job_id`; the work runs in
the background, one job at a time on the GPU.

1. Submit a file to `/transcribe`, `/separate`, `/transcribe-only` or `/correct`.
2. Poll `GET /jobs/{job_id}` until `status` is `completed` or `failed`.
3. Download the result from `GET /jobs/{job_id}/result`, then optionally
   `DELETE /jobs/{job_id}`. Finished jobs are removed automatically after
   `JOB_TTL_SECONDS`.

## Batch processing

`POST /batch` transcribes many songs as **one batch** — a folder (e.g. an
album) or a CSV listing file paths — reading them straight from storage
mounted into the container. The batch keeps running on the server regardless
of the client; poll `GET /batches/{batch_id}` for per-song progress.

Batch mode is off until **`TRANSCRIBER_ALLOWED_ROOTS`** lists the mounted
directories it may read from (comma-separated). Every path is fully resolved
(`..`, symlinks) and must stay inside those roots. **`TRANSCRIBER_BATCH_OUTPUT`**
decides where results go: `download` (default — keep them here for download)
or `alongside` (also write `<song>.lrc` next to each song; needs write access).

## Correction threshold

When the Genius lyrics match the transcription worse than
**`CORRECTION_MIN_MATCH`** percent (default `0` = always correct), the Genius
text is discarded, the Whisper transcription is kept, and the job gets a
warning. Override per request with `correction_min_match`.

## Errors

Errors use FastAPI's `{"detail": ...}` body. Errors added with batch mode carry
a machine-readable code: `{"detail": {"code": "path_not_allowed", "message": ..., ...}}`.
"""

OPENAPI_TAGS = [
    {"name": "Health", "description": "Service, GPU and configuration status."},
    {"name": "Processing", "description": "Submit a single uploaded file as a background job."},
    {"name": "Batch", "description": "Process many songs from mounted storage as one batch."},
    {"name": "Jobs", "description": "Poll, download and delete jobs."},
]


# ── Form fields ─────────────────────────────────────────────────────────────


def audio_file_field(what: str = "Audio file to process"):
    return File(..., description=f"{what}. Any format ffmpeg can decode (FLAC, MP3, M4A, WAV, ...).")


def format_field():
    return Form(
        OutputFormat.lrc,
        description="Output format: `lrc` (timestamped lyrics), `txt` (plain text) or `all` (both, returned as ZIP).",
    )


def no_separation_field():
    return Form(False, description="Skip vocal separation — use when the file already contains isolated vocals.")


def demucs_model_field():
    return Form(
        DEFAULT_DEMUCS_MODEL,
        description=f"Demucs model used for vocal separation (default `{DEFAULT_DEMUCS_MODEL}`, set by `DEMUCS_MODEL`).",
    )


def whisper_model_field():
    return Form(
        "large-v3-turbo",
        description=f"Whisper model: {WHISPER_MODELS}. Larger models are more accurate but need more VRAM — see `whisper_model_fit` in `/health`.",
    )


def language_field():
    return Form(None, description="Force a language code such as `de` or `en`. Empty = auto-detect.")


def artist_field(required: bool = False):
    if required:
        return Form(..., description="Artist name used for the Genius lookup.")
    return Form(None, description="Artist for the Genius lookup; overrides the audio file's tags.")


def title_field(required: bool = False):
    if required:
        return Form(..., description="Song title used for the Genius lookup.")
    return Form(None, description="Song title for the Genius lookup; overrides the audio file's tags.")


def no_correction_field():
    return Form(False, description="Skip the Genius lyrics correction and return the raw Whisper transcription.")


def correction_min_match_field():
    return Form(
        None,
        ge=0,
        le=100,
        description=(
            "Minimum Genius match in percent (0–100). Below it the Genius lyrics are discarded and the "
            "Whisper transcription is kept, with a job warning. Empty = `CORRECTION_MIN_MATCH` (default 0)."
        ),
    )


# ── Responses ───────────────────────────────────────────────────────────────


def _error(description: str) -> dict:
    return {"model": ErrorResponse, "description": description}


RESP_400_NO_FILENAME = {400: _error("The uploaded file has no filename.")}
RESP_404_JOB = {404: _error("Job not found (unknown id, deleted, or expired after `JOB_TTL_SECONDS`).")}
RESP_409_JOB_STATE = {409: _error("The job is not finished yet (`pending` or `processing`).")}
RESP_409_JOB_DELETE = {
    409: _error("The job is still processing, or belongs to a batch (`job_in_batch` — delete the batch instead).")
}
RESP_404_BATCH = {404: _error("Batch not found (`batch_not_found`): unknown id, deleted, or expired.")}
RESP_BATCH_PATH_ERRORS = {
    403: _error("`batch_disabled` (no `TRANSCRIBER_ALLOWED_ROOTS` set) or `path_not_allowed` (path outside the allowed roots)."),
    404: _error("`path_not_found`: the directory does not exist."),
    422: _error(
        "`invalid_batch_request` (neither or both of `directory`/`csv`, no audio files, more than 5000 songs), "
        "`invalid_csv` (with `errors: [{line, path, reason}]`), or a form field out of range."
    ),
}

_BINARY = {"schema": {"type": "string", "format": "binary"}}

RESP_RESULT_FILE = {
    200: {
        "description": (
            "A single result file (`.lrc`, `.txt`, or the vocals `.wav` from `/separate`) as "
            "`application/octet-stream`, or several files as `application/zip`."
        ),
        "content": {"application/octet-stream": _BINARY, "application/zip": _BINARY},
    },
}
RESP_204_NO_FILES = {204: {"description": "The job finished without output files (e.g. no vocals detected)."}}
RESP_BATCH_ZIP = {
    200: {
        "description": "ZIP with one folder per song (`<song>/<song>.lrc`); repeated song names get ` (2)`, ` (3)`.",
        "content": {"application/zip": _BINARY},
    },
    204: {"description": "No song of the batch has produced files yet."},
}
