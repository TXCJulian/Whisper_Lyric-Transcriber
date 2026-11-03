import os
import io
import zipfile
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, Response
from fastapi.responses import FileResponse, StreamingResponse

from app.models import (
    OutputFormat,
    JobResponse,
    JobStatusResponse,
    JobResultFile,
    JobResultResponse,
    DeleteResponse,
    HealthResponse,
)
from app.job_manager import JobInBatch, JobManager
from app.pipeline import run_full_pipeline
from app.separation import (
    separate_vocals,
    load_model as load_separation_model,
)
from app.transcription_engine import get_engine
from app.transcription import write_output_files
from app.correction import (
    fetch_genius_lyrics,
    correct_transcription,
    get_metadata_from_file,
)
from app.gpu_backend import (
    get_backend,
    get_device_name,
    get_vram_info,
    get_whisper_model_fit,
    use_faster_whisper,
)
from app import api_docs as docs
from app import batch_routes, config
from app.batch_manager import BatchManager
from app.runtime_profile import get_runtime_profile
from app.version import get_version_info

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)
# Quiet the per-file HTTP request noise from model downloads (huggingface_hub
# uses httpx; older releases used requests via urllib3) -- keep warnings/errors.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

job_manager = JobManager(max_workers=1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    info = get_version_info()
    logger.info(
        f"Lyric Transcriber {info['version']} (build {info['build']}, "
        f"commit {info['commit'][:7]}, image {info['image_variant']})"
    )
    backend = get_backend()
    device_name = get_device_name()
    logger.info(f"GPU backend: {backend} ({device_name})")

    engine = get_engine()
    engine_name = "faster-whisper" if use_faster_whisper() else "openai-whisper"
    logger.info(f"Transcription engine: {engine_name}")

    if os.getenv("PRELOAD_MODELS", "").lower() == "true":
        logger.info("Preloading models...")
        load_separation_model()
        engine.load_model()
        logger.info("Models preloaded")

    yield
    logger.info("Shutting down lyric-transcriber service")


app = FastAPI(
    title="Lyric Transcriber API",
    version=get_version_info()["version"],
    description=docs.API_DESCRIPTION,
    openapi_tags=docs.OPENAPI_TAGS,
    lifespan=lifespan,
)

batch_routes.batch_manager = BatchManager(job_manager)
app.include_router(batch_routes.router)


# ── Health ──────────────────────────────────────────────────────────────────


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Service status",
    description=(
        "Running version/build/commit, the environment chosen at container start, GPU and "
        "transcription engine, VRAM with per-model fit, the default correction threshold and "
        "whether batch mode is enabled."
    ),
)
def health():
    vram = get_vram_info()
    return {
        "status": "ok",
        "version": get_version_info(),
        "runtime_profile": get_runtime_profile() or None,
        "gpu_backend": get_backend(),
        "gpu_name": get_device_name(),
        "transcription_engine": "faster-whisper" if use_faster_whisper() else "openai-whisper",
        "vram_total_mb": vram["total_mb"] if vram else None,
        "whisper_model_fit": get_whisper_model_fit(vram),
        "correction_min_match": config.CORRECTION_MIN_MATCH,
        "batch": {
            "enabled": bool(config.ALLOWED_ROOTS),
            "allowed_roots": [str(root) for root in config.ALLOWED_ROOTS],
            "output_mode": config.BATCH_OUTPUT,
        },
    }


# ── Full Pipeline ──────────────────────────────────────────────────────────


@app.post(
    "/transcribe",
    response_model=JobResponse,
    tags=["Processing"],
    summary="Full pipeline: separate, transcribe, correct",
    description=(
        "Isolates the vocals with Demucs, transcribes them with Whisper (word timestamps, VAD) "
        "and corrects the text against Genius lyrics. Artist/title for Genius come from the "
        "file's tags unless given. Returns a `job_id`; poll `GET /jobs/{job_id}`."
    ),
    responses={**docs.RESP_400_NO_FILENAME},
)
async def submit_transcribe(
    file: UploadFile = docs.audio_file_field(),
    format: OutputFormat = docs.format_field(),
    no_separation: bool = docs.no_separation_field(),
    demucs_model: str = docs.demucs_model_field(),
    whisper_model: str = docs.whisper_model_field(),
    language: str | None = docs.language_field(),
    artist: str | None = docs.artist_field(),
    title: str | None = docs.title_field(),
    no_correction: bool = docs.no_correction_field(),
    correction_min_match: float | None = docs.correction_min_match_field(),
):
    # Normalize empty strings to None (multipart forms send "" for omitted fields)
    language = language if language else None
    artist = artist if artist else None
    title = title if title else None

    if not file.filename:
        raise HTTPException(status_code=400, detail="File must have a filename")

    job = job_manager.create_job()
    input_path = os.path.join(job.input_dir, file.filename)
    with open(input_path, "wb") as f:
        content = await file.read()
        f.write(content)

    job_manager.submit(
        job,
        run_full_pipeline,
        input_path=input_path,
        output_dir=job.output_dir,
        format=format.value,
        no_separation=no_separation,
        demucs_model=demucs_model,
        whisper_model=whisper_model,
        language=language,
        artist=artist,
        title=title,
        no_correction=no_correction,
        correction_min_match=correction_min_match,
    )
    return JobResponse(job_id=job.id)


# ── Vocal Separation Only ──────────────────────────────────────────────────


@app.post(
    "/separate",
    response_model=JobResponse,
    tags=["Processing"],
    summary="Vocal separation only",
    description="Isolates the vocals with Demucs. The job result is the vocals as a `.wav` file.",
    responses={**docs.RESP_400_NO_FILENAME},
)
async def submit_separate(
    file: UploadFile = docs.audio_file_field(),
    demucs_model: str = docs.demucs_model_field(),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="File must have a filename")

    job = job_manager.create_job()
    input_path = os.path.join(job.input_dir, file.filename)
    with open(input_path, "wb") as f:
        content = await file.read()
        f.write(content)

    def task(progress_callback, warning_callback, **kwargs):
        progress_callback("Separating vocals...")
        result_path = separate_vocals(
            kwargs["input_path"], kwargs["output_dir"], model_name=kwargs["demucs_model"]
        )
        progress_callback("Complete")
        return [result_path]

    job_manager.submit(
        job,
        task,
        input_path=input_path,
        output_dir=job.output_dir,
        demucs_model=demucs_model,
    )
    return JobResponse(job_id=job.id)


# ── Transcription Only ─────────────────────────────────────────────────────


@app.post(
    "/transcribe-only",
    response_model=JobResponse,
    tags=["Processing"],
    summary="Transcription only",
    description=(
        "Transcribes the file as-is with Whisper — no vocal separation, no Genius correction. "
        "Best for files that already contain isolated vocals."
    ),
    responses={**docs.RESP_400_NO_FILENAME},
)
async def submit_transcribe_only(
    file: UploadFile = docs.audio_file_field("Audio file, ideally isolated vocals"),
    format: OutputFormat = docs.format_field(),
    whisper_model: str = docs.whisper_model_field(),
    language: str | None = docs.language_field(),
):
    language = language if language else None

    if not file.filename:
        raise HTTPException(status_code=400, detail="File must have a filename")

    job = job_manager.create_job()
    input_path = os.path.join(job.input_dir, file.filename)
    with open(input_path, "wb") as f:
        content = await file.read()
        f.write(content)

    def task(progress_callback, warning_callback, **kwargs):
        progress_callback("Transcribing...")
        engine = get_engine()
        segments, _ = engine.transcribe(
            kwargs["input_path"],
            model_size=kwargs["whisper_model"],
            language=kwargs["language"],
            language_callback=lambda lang: progress_callback(f"Detected language: {lang}"),
        )

        # Check if any vocals were detected
        has_content = any(seg.text.strip() for seg in segments)
        if not has_content:
            logger.warning("No vocals detected in audio, skipping output file creation")
            warning_callback("No vocals detected in audio")
            progress_callback("Complete - no vocals detected")
            return []

        stem = os.path.splitext(os.path.basename(kwargs["input_path"]))[0]
        output_files = write_output_files(
            segments, kwargs["output_dir"], stem, kwargs["format"]
        )
        progress_callback("Complete")
        return output_files

    job_manager.submit(
        job,
        task,
        input_path=input_path,
        output_dir=job.output_dir,
        whisper_model=whisper_model,
        language=language,
        format=format.value,
    )
    return JobResponse(job_id=job.id)


# ── Lyrics Correction Only ─────────────────────────────────────────────────


@app.post(
    "/correct",
    response_model=JobResponse,
    tags=["Processing"],
    summary="Transcribe and correct (no separation)",
    description=(
        "Transcribes the file without vocal separation, then corrects it against the Genius "
        "lyrics of the given artist and title. The job fails if Genius has no lyrics for them."
    ),
    responses={**docs.RESP_400_NO_FILENAME},
)
async def submit_correct(
    file: UploadFile = docs.audio_file_field("Audio or isolated vocals"),
    artist: str = docs.artist_field(required=True),
    title: str = docs.title_field(required=True),
    format: OutputFormat = docs.format_field(),
    correction_min_match: float | None = docs.correction_min_match_field(),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="File must have a filename")

    job = job_manager.create_job()
    input_path = os.path.join(job.input_dir, file.filename)
    with open(input_path, "wb") as f:
        content = await file.read()
        f.write(content)

    def task(progress_callback, warning_callback, **kwargs):
        progress_callback("Transcribing for correction...")
        engine = get_engine()
        segments, _ = engine.transcribe(
            kwargs["input_path"],
            artist=kwargs["artist"],
            title=kwargs["title"],
            language_callback=lambda lang: progress_callback(f"Detected language: {lang}"),
        )

        # Check if any vocals were detected
        has_content = any(seg.text.strip() for seg in segments)
        if not has_content:
            logger.warning("No vocals detected in audio, skipping output file creation")
            warning_callback("No vocals detected in audio")
            progress_callback("Complete - no vocals detected")
            return []

        progress_callback("Fetching Genius lyrics...")
        genius_lyrics = fetch_genius_lyrics(kwargs["artist"], kwargs["title"])
        if not genius_lyrics:
            raise ValueError(
                f"No lyrics found on Genius for '{kwargs['artist']} - {kwargs['title']}'"
            )

        progress_callback("Applying correction...")
        corrected = correct_transcription(
            segments,
            genius_lyrics,
            min_match=config.resolve_min_match(kwargs["correction_min_match"]),
            warning_callback=warning_callback,
        )

        stem = os.path.splitext(os.path.basename(kwargs["input_path"]))[0]
        output_files = write_output_files(
            corrected, kwargs["output_dir"], stem, kwargs["format"]
        )
        progress_callback("Complete")
        return output_files

    job_manager.submit(
        job,
        task,
        input_path=input_path,
        output_dir=job.output_dir,
        artist=artist,
        title=title,
        format=format.value,
        correction_min_match=correction_min_match,
    )
    return JobResponse(job_id=job.id)


# ── Job Status & Results ───────────────────────────────────────────────────


@app.get(
    "/jobs/{job_id}",
    response_model=JobStatusResponse,
    tags=["Jobs"],
    summary="Job status",
    description="Status, latest progress message, error and warnings of a job. Poll until `completed` or `failed`.",
    responses={**docs.RESP_404_JOB},
)
def get_job_status(job_id: str):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatusResponse(
        job_id=job.id,
        status=job.status,
        progress=job.progress,
        error=job.error,
        warning=job.warning,
        warnings=list(job.warnings),
        created_at=job.created_at,
        completed_at=job.completed_at,
    )


@app.get(
    "/jobs/{job_id}/result",
    tags=["Jobs"],
    summary="Download job results",
    description="A single result file is returned directly, several files (e.g. `format=all`) as a ZIP.",
    responses={
        **docs.RESP_RESULT_FILE,
        **docs.RESP_204_NO_FILES,
        **docs.RESP_404_JOB,
        **docs.RESP_409_JOB_STATE,
    },
)
def get_job_result(job_id: str):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "completed":
        raise HTTPException(status_code=409, detail=f"Job is {job.status.value}, not completed")

    output_files: list[str] = job.result or []
    existing = [f for f in output_files if os.path.isfile(f)]

    if not existing:
        # No files created - successful processing but no output (e.g., no vocals detected)
        return Response(status_code=204)

    if len(existing) == 1:
        return FileResponse(
            existing[0],
            filename=os.path.basename(existing[0]),
            media_type="application/octet-stream",
        )

    # Multiple files: return as ZIP
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for filepath in existing:
            zf.write(filepath, os.path.basename(filepath))
    zip_buffer.seek(0)

    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="job_{job_id}_result.zip"'},
    )


@app.get(
    "/jobs/{job_id}/result/info",
    response_model=JobResultResponse,
    tags=["Jobs"],
    summary="List result files",
    description="Names and formats of a finished job's result files, without downloading them.",
    responses={**docs.RESP_404_JOB, **docs.RESP_409_JOB_STATE},
)
def get_job_result_info(job_id: str):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "completed":
        raise HTTPException(status_code=409, detail=f"Job is {job.status.value}, not completed")

    output_files: list[str] = job.result or []
    files = []
    for f in output_files:
        if os.path.isfile(f):
            ext = os.path.splitext(f)[1].lstrip(".")
            files.append(JobResultFile(filename=os.path.basename(f), format=ext))

    return JobResultResponse(job_id=job.id, files=files)


@app.delete(
    "/jobs/{job_id}",
    response_model=DeleteResponse,
    tags=["Jobs"],
    summary="Delete a job",
    description=(
        "Removes a finished job and its files. Songs of a batch can't be deleted one by one — "
        "delete the batch instead."
    ),
    responses={**docs.RESP_404_JOB, **docs.RESP_409_JOB_DELETE},
)
def delete_job(job_id: str):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status == "processing":
        raise HTTPException(status_code=409, detail="Cannot delete a processing job")
    try:
        deleted = job_manager.delete_job(job_id)
    except JobInBatch as e:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "job_in_batch",
                "message": f"Job belongs to batch {e.batch_id}; delete the batch instead",
                "batch_id": e.batch_id,
            },
        )
    if not deleted:
        raise HTTPException(status_code=500, detail="Failed to delete job")
    return {"status": "deleted", "job_id": job_id}
