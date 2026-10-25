"""HTTP endpoints for batch transcription (songs read from mounted shares)."""
import os
from collections import Counter

from fastapi import APIRouter, File, Form, HTTPException, Response, UploadFile
from fastapi.concurrency import run_in_threadpool

from app import api_docs as docs
from app.batch_input import BatchInputError, collect_batch_paths
from app.batch_manager import Batch, BatchFinished, BatchManager
from app.models import (
    BatchCounts,
    BatchCreatedResponse,
    BatchDeleteResponse,
    BatchItemResponse,
    BatchStatusResponse,
    BatchSummary,
    ErrorDetail,
    OutputFormat,
)
from app.paths import PathError

router = APIRouter(tags=["Batch"])

# Set by app.main at startup (tests replace it).
batch_manager: BatchManager | None = None


def _error(status_code: int, code: str, message: str, **extra) -> HTTPException:
    detail = ErrorDetail(code=code, message=message, **extra)
    return HTTPException(status_code=status_code, detail=detail.model_dump(exclude_none=True))


def _get_batch(batch_id: str) -> Batch:
    batch = batch_manager.get(batch_id)
    if batch is None:
        raise _error(404, "batch_not_found", "Batch not found", batch_id=batch_id)
    return batch


def _counts(batch: Batch) -> BatchCounts:
    return BatchCounts(**Counter(item.status.value for item in batch.items))


def _summary(batch: Batch) -> BatchSummary:
    return BatchSummary(
        batch_id=batch.id,
        status=batch.status,
        created_at=batch.created_at,
        completed_at=batch.completed_at,
        total=len(batch.items),
        counts=_counts(batch),
    )


def _status(batch: Batch) -> BatchStatusResponse:
    items = []
    for item in batch.items:
        job = batch_manager.job_manager.get_job(item.job_id) if item.job_id else None
        items.append(BatchItemResponse(
            path=item.path,
            status=item.status,
            job_id=item.job_id,
            progress=job.progress if job else None,
            error=item.error,
            warnings=list(job.warnings) if job else [],
            outputs=[os.path.basename(f) for f in (job.result or [])] if job else [],
            written_alongside=list(item.written_alongside),
        ))
    return BatchStatusResponse(
        **_summary(batch).model_dump(), output_mode=batch.output_mode, items=items
    )


@router.post(
    "/batch",
    response_model=BatchCreatedResponse,
    status_code=202,
    summary="Start a batch",
    description=(
        "Transcribes every song of a folder, or every file listed in a CSV, as one batch. Songs "
        "are read from storage mounted into the container and must lie inside "
        "`TRANSCRIBER_ALLOWED_ROOTS`. Songs run one after another; options apply to every song, "
        "artist/title for Genius come from each file's tags. Poll `GET /batches/{batch_id}`."
    ),
    responses={**docs.RESP_BATCH_PATH_ERRORS},
)
async def create_batch(
    directory: str | None = Form(
        None,
        description="Folder whose audio files form the batch (e.g. `/music/Artist/Album`). Exactly one of `directory` or `csv`.",
    ),
    csv: UploadFile | None = File(
        None,
        description=(
            "CSV with one absolute file path per row (first column; an optional `path` header, blank "
            "lines and `#` comments are ignored; quote paths that contain commas). Exactly one of "
            "`directory` or `csv`. Any invalid line rejects the whole request."
        ),
    ),
    recursive: bool = Form(False, description="With `directory`: include audio files in subfolders."),
    format: OutputFormat = docs.format_field(),
    no_separation: bool = docs.no_separation_field(),
    demucs_model: str = docs.demucs_model_field(),
    whisper_model: str = docs.whisper_model_field(),
    language: str | None = docs.language_field(),
    no_correction: bool = docs.no_correction_field(),
    correction_min_match: float | None = docs.correction_min_match_field(),
    skip_existing: bool = Form(
        True,
        description=(
            "`alongside` mode only: songs whose lyrics files already exist next to them are marked "
            "`skipped` and never overwritten (checked at start and again before writing)."
        ),
    ),
):
    csv_content = await csv.read() if csv is not None else None
    try:
        # Listing a large NAS folder can take a while; keep it off the event loop
        # so /health and job polling stay responsive.
        paths = await run_in_threadpool(
            collect_batch_paths, directory or None, csv_content, recursive
        )
    except PathError as e:
        raise _error(e.status_code, e.code, e.message, path=e.path)
    except BatchInputError as e:
        raise _error(e.status_code, e.code, e.message, errors=e.errors or None)

    batch = batch_manager.create_batch(paths, {
        "format": format.value,
        "no_separation": no_separation,
        "demucs_model": demucs_model,
        "whisper_model": whisper_model,
        "language": language or None,
        "no_correction": no_correction,
        "correction_min_match": correction_min_match,
        "skip_existing": skip_existing,
    })
    return BatchCreatedResponse(batch_id=batch.id, total=len(batch.items))


@router.get(
    "/batches",
    response_model=list[BatchSummary],
    summary="List batches",
    description="All batches still held by the service, newest first, with per-status song counts.",
)
def list_batches():
    return [_summary(batch) for batch in batch_manager.list_batches()]


@router.get(
    "/batches/{batch_id}",
    response_model=BatchStatusResponse,
    summary="Batch status",
    description=(
        "Overall status and per-song status, progress, warnings and result files. The batch is "
        "`completed` when all songs are done (individual songs may have failed), `failed` only if "
        "every song failed, `cancelled` after a cancel."
    ),
    responses={**docs.RESP_404_BATCH},
)
def get_batch(batch_id: str):
    return _status(_get_batch(batch_id))


@router.get(
    "/batches/{batch_id}/result",
    summary="Download batch results",
    description="Every finished song's result files as one ZIP. Works while the batch is still running.",
    responses={**docs.RESP_BATCH_ZIP, **docs.RESP_404_BATCH},
)
def get_batch_result(batch_id: str):
    _get_batch(batch_id)
    content = batch_manager.zip_results(batch_id)
    if content is None:
        return Response(status_code=204)
    return Response(
        content=content,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="batch_{batch_id}.zip"'},
    )


@router.post(
    "/batches/{batch_id}/cancel",
    response_model=BatchStatusResponse,
    summary="Cancel a batch",
    description=(
        "Stops the batch but keeps it: waiting songs are cancelled, a song that is already running "
        "finishes. Finished songs' results stay downloadable until the batch expires."
    ),
    responses={**docs.RESP_404_BATCH, 409: {"model": docs.ErrorResponse, "description": "`batch_finished`: the batch has already ended."}},
)
def cancel_batch(batch_id: str):
    batch = _get_batch(batch_id)
    try:
        batch_manager.cancel(batch_id)
    except BatchFinished:
        raise _error(409, "batch_finished", "Batch has already ended", batch_id=batch_id)
    return _status(batch)


@router.delete(
    "/batches/{batch_id}",
    response_model=BatchDeleteResponse,
    summary="Delete a batch",
    description=(
        "Cancels the batch and removes it with all its results. Files already written next to "
        "songs (`alongside` mode) are kept."
    ),
    responses={
        200: {"description": "Deleted."},
        202: {"model": BatchDeleteResponse, "description": "A song is still running; the batch is deleted when it finishes (`status: cancelling`)."},
        **docs.RESP_404_BATCH,
    },
)
def delete_batch(batch_id: str, response: Response):
    _get_batch(batch_id)
    status = batch_manager.delete(batch_id)
    if status == "cancelling":
        response.status_code = 202
    return BatchDeleteResponse(status=status, batch_id=batch_id)
