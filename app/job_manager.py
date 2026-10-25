import os
import uuid
import shutil
import threading
import logging
from datetime import datetime, timezone
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable

from .models import JobStatus

logger = logging.getLogger(__name__)

JOBS_BASE_DIR = os.getenv("JOBS_DIR", "/app/jobs")
JOB_TTL_SECONDS = int(os.getenv("JOB_TTL_SECONDS", "3600"))  # 1 hour default
MAX_COMPLETED_JOBS = 200


@dataclass
class Job:
    id: str
    status: JobStatus = JobStatus.pending
    progress: str | None = None
    error: str | None = None
    warning: str | None = None
    # Every warning the task reported; `warning` keeps only the last one, as
    # the API has always returned.
    warnings: list[str] = field(default_factory=list)
    # Set for songs of a batch: the batch owns the job's lifetime.
    batch_id: str | None = None
    result: Any = None
    created_at: str = ""
    completed_at: str | None = None
    task_fn: Callable | None = field(default=None, repr=False)
    task_kwargs: dict = field(default_factory=dict, repr=False)

    @property
    def input_dir(self) -> str:
        return os.path.join(JOBS_BASE_DIR, self.id, "input")

    @property
    def output_dir(self) -> str:
        return os.path.join(JOBS_BASE_DIR, self.id, "output")


class JobInBatch(Exception):
    """Raised when deleting a job that belongs to a batch."""

    def __init__(self, batch_id: str):
        super().__init__(f"Job belongs to batch {batch_id}")
        self.batch_id = batch_id


class JobManager:
    def __init__(self, max_workers: int = 1):
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._lock = threading.Lock()
        self._semaphore = threading.Semaphore(max_workers)
        # JOBS_BASE_DIR is created with the first job (create_job), not at
        # import -- importing the app must not need write access to it.

    def create_job(self, batch_id: str | None = None) -> Job:
        """Create a new job with its directories. Call submit() or run() to start it."""
        job_id = uuid.uuid4().hex[:12]
        job = Job(
            id=job_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            batch_id=batch_id,
        )
        os.makedirs(job.input_dir, exist_ok=True)
        os.makedirs(job.output_dir, exist_ok=True)
        with self._lock:
            self._jobs[job_id] = job
        return job

    def submit(self, job: Job, task_fn: Callable, **kwargs):
        """Submit a created job for processing."""
        thread = threading.Thread(
            target=self.run, args=(job, task_fn), kwargs=kwargs, daemon=True
        )
        thread.start()

    def run(self, job: Job, task_fn: Callable, **kwargs):
        """Run a created job in the calling thread (waits for the GPU slot)."""
        job.task_fn = task_fn
        job.task_kwargs = kwargs
        self._semaphore.acquire()
        try:
            if job.task_fn is None:
                raise RuntimeError(f"Job {job.id} has no task function")
            
            with self._lock:
                job.status = JobStatus.processing

            def progress_callback(msg: str):
                with self._lock:
                    job.progress = msg
            
            def warning_callback(msg: str):
                with self._lock:
                    job.warning = msg
                    job.warnings.append(msg)

            result = job.task_fn(
                progress_callback=progress_callback,
                warning_callback=warning_callback,
                **job.task_kwargs
            )

            with self._lock:
                job.status = JobStatus.completed
                job.result = result
                job.completed_at = datetime.now(timezone.utc).isoformat()

            logger.info(f"Job {job.id} completed")
        except Exception as e:
            logger.exception(f"Job {job.id} failed")
            with self._lock:
                job.status = JobStatus.failed
                job.error = str(e)
                job.completed_at = datetime.now(timezone.utc).isoformat()
        finally:
            # Clean up task references to free memory
            job.task_fn = None
            job.task_kwargs = {}
            self._semaphore.release()
            self._evict_old_jobs()

    def get_job(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def delete_job(self, job_id: str, force: bool = False) -> bool:
        """Delete a job and its files. Returns True if found and deleted.

        Jobs owned by a batch raise JobInBatch unless *force* (used by the
        batch manager when it removes the whole batch).
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None and job.batch_id and not force:
                raise JobInBatch(job.batch_id)
            job = self._jobs.pop(job_id, None)
        if job is None:
            return False
        # Don't delete processing jobs
        if job.status == "processing":
            with self._lock:
                self._jobs[job_id] = job
            return False
        job_dir = os.path.join(JOBS_BASE_DIR, job_id)
        if os.path.isdir(job_dir):
            shutil.rmtree(job_dir, ignore_errors=True)
        return True

    def _evict_old_jobs(self):
        """Remove old completed/failed jobs beyond the limit."""
        now = datetime.now(timezone.utc)
        to_remove = []

        with self._lock:
            for job_id, job in self._jobs.items():
                if job.status not in ("completed", "failed") or job.batch_id:
                    continue
                if job.completed_at:
                    completed = datetime.fromisoformat(job.completed_at)
                    age = (now - completed).total_seconds()
                    if age > JOB_TTL_SECONDS:
                        to_remove.append(job_id)

            # Also enforce max count
            completed_jobs = [
                j for j in self._jobs.values()
                if j.status in ("completed", "failed") and not j.batch_id
            ]
            if len(completed_jobs) > MAX_COMPLETED_JOBS:
                excess = completed_jobs[: len(completed_jobs) - MAX_COMPLETED_JOBS]
                for j in excess:
                    if j.id not in to_remove:
                        to_remove.append(j.id)

            for job_id in to_remove:
                self._jobs.pop(job_id, None)

        # Clean up files outside lock
        for job_id in to_remove:
            job_dir = os.path.join(JOBS_BASE_DIR, job_id)
            if os.path.isdir(job_dir):
                shutil.rmtree(job_dir, ignore_errors=True)
            logger.info(f"Evicted old job {job_id}")
