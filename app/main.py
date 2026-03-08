import os
import io
import zipfile
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Response
from fastapi.responses import FileResponse, StreamingResponse
from dotenv import load_dotenv

from app.models import (
    OutputFormat,
    JobResponse,
    JobStatusResponse,
    JobResultFile,
    JobResultResponse,
)
from app.job_manager import JobManager
from app.pipeline import run_full_pipeline
from app.separation import separate_vocals, load_model as load_separation_model
from app.transcription_engine import get_engine
from app.transcription import write_output_files
from app.correction import (
    fetch_genius_lyrics,
    correct_transcription,
    get_metadata_from_file,
)
from app.gpu_backend import get_backend, get_device_name, use_faster_whisper

load_dotenv()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

job_manager = JobManager(max_workers=1)


@asynccontextmanager
async def lifespan(app: FastAPI):
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


app = FastAPI(title="Lyric Transcriber API", version="1.0.0", lifespan=lifespan)


# ── Health ──────────────────────────────────────────────────────────────────


@app.get("/health")
def health():
    return {
        "status": "ok",
        "gpu_backend": get_backend(),
        "gpu_name": get_device_name(),
        "transcription_engine": "faster-whisper" if use_faster_whisper() else "openai-whisper",
    }


# ── Full Pipeline ──────────────────────────────────────────────────────────


@app.post("/transcribe", response_model=JobResponse)
async def submit_transcribe(
    file: UploadFile = File(...),
    format: OutputFormat = Form(OutputFormat.lrc),
    no_separation: bool = Form(False),
    whisper_model: str = Form("large-v3-turbo"),
    language: str | None = Form(None),
    artist: str | None = Form(None),
    title: str | None = Form(None),
    no_correction: bool = Form(False),
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
        whisper_model=whisper_model,
        language=language,
        artist=artist,
        title=title,
        no_correction=no_correction,
    )
    return JobResponse(job_id=job.id)


# ── Vocal Separation Only ──────────────────────────────────────────────────


@app.post("/separate", response_model=JobResponse)
async def submit_separate(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="File must have a filename")

    job = job_manager.create_job()
    input_path = os.path.join(job.input_dir, file.filename)
    with open(input_path, "wb") as f:
        content = await file.read()
        f.write(content)

    def task(progress_callback, warning_callback, **kwargs):
        progress_callback("Separating vocals...")
        result_path = separate_vocals(kwargs["input_path"], kwargs["output_dir"])
        progress_callback("Complete")
        return [result_path]

    job_manager.submit(job, task, input_path=input_path, output_dir=job.output_dir)
    return JobResponse(job_id=job.id)


# ── Transcription Only ─────────────────────────────────────────────────────


@app.post("/transcribe-only", response_model=JobResponse)
async def submit_transcribe_only(
    file: UploadFile = File(...),
    format: OutputFormat = Form(OutputFormat.lrc),
    whisper_model: str = Form("large-v3-turbo"),
    language: str | None = Form(None),
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


@app.post("/correct", response_model=JobResponse)
async def submit_correct(
    file: UploadFile = File(...),
    artist: str = Form(...),
    title: str = Form(...),
    format: OutputFormat = Form(OutputFormat.lrc),
):
    """Correct an existing transcription using Genius lyrics.

    Accepts an audio file (to re-transcribe and then correct) or a pre-existing
    vocals/audio file that will be transcribed first, then corrected.
    """
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
        corrected = correct_transcription(segments, genius_lyrics)

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
    )
    return JobResponse(job_id=job.id)


# ── Job Status & Results ───────────────────────────────────────────────────


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
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
        created_at=job.created_at,
        completed_at=job.completed_at,
    )


@app.get("/jobs/{job_id}/result")
def get_job_result(job_id: str):
    """Download job results. Single file returned directly, multiple as ZIP. Returns 204 if no files."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "completed":
        raise HTTPException(status_code=409, detail=f"Job is {job.status}, not completed")

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


@app.get("/jobs/{job_id}/result/info", response_model=JobResultResponse)
def get_job_result_info(job_id: str):
    """Get metadata about result files without downloading them."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "completed":
        raise HTTPException(status_code=409, detail=f"Job is {job.status}, not completed")

    output_files: list[str] = job.result or []
    files = []
    for f in output_files:
        if os.path.isfile(f):
            ext = os.path.splitext(f)[1].lstrip(".")
            files.append(JobResultFile(filename=os.path.basename(f), format=ext))

    return JobResultResponse(job_id=job.id, files=files)


@app.delete("/jobs/{job_id}")
def delete_job(job_id: str):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status == "processing":
        raise HTTPException(status_code=409, detail="Cannot delete a processing job")
    if not job_manager.delete_job(job_id):
        raise HTTPException(status_code=500, detail="Failed to delete job")
    return {"status": "deleted", "job_id": job_id}
