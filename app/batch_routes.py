"""HTTP endpoints for batch transcription (songs read from mounted shares)."""
import os
from collections import Counter

from fastapi import APIRouter, File, Form, HTTPException, Response, UploadFile
from fastapi.concurrency import run_in_threadpool

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
from app.separation import DEFAULT_DEMUCS_MODEL

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


@router.post("/batch", response_model=BatchCreatedResponse, status_code=202)
async def create_batch(
    directory: str | None = Form(None),
    csv: UploadFile | None = File(None),
    recursive: bool = Form(False),
    format: OutputFormat = Form(OutputFormat.lrc),
    no_separation: bool = Form(False),
    demucs_model: str = Form(DEFAULT_DEMUCS_MODEL),
    whisper_model: str = Form("large-v3-turbo"),
    language: str | None = Form(None),
    no_correction: bool = Form(False),
    correction_min_match: float | None = Form(None, ge=0, le=100),
    skip_existing: bool = Form(True),
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


@router.get("/batches", response_model=list[BatchSummary])
def list_batches():
    return [_summary(batch) for batch in batch_manager.list_batches()]


@router.get("/batches/{batch_id}", response_model=BatchStatusResponse)
def get_batch(batch_id: str):
    return _status(_get_batch(batch_id))


@router.get("/batches/{batch_id}/result")
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


@router.post("/batches/{batch_id}/cancel", response_model=BatchStatusResponse)
def cancel_batch(batch_id: str):
    batch = _get_batch(batch_id)
    try:
        batch_manager.cancel(batch_id)
    except BatchFinished:
        raise _error(409, "batch_finished", "Batch has already ended", batch_id=batch_id)
    return _status(batch)


@router.delete("/batches/{batch_id}", response_model=BatchDeleteResponse)
def delete_batch(batch_id: str, response: Response):
    _get_batch(batch_id)
    status = batch_manager.delete(batch_id)
    if status == "cancelling":
        response.status_code = 202
    return BatchDeleteResponse(status=status, batch_id=batch_id)
