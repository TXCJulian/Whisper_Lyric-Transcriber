"""Batches: many songs processed as one unit.

Each song is a regular job (owned by its batch, see JobManager.batch_id), so
per-song status and results stay available through /jobs/{id}. One runner
thread per batch processes the songs in order; every song still goes through
JobManager's GPU semaphore, so batch songs and single requests never run on
the GPU at the same time.
"""
import io
import logging
import os
import shutil
import threading
import uuid
import zipfile
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from app import config
from app import job_manager as jm
from app.models import BatchStatus, ItemStatus, JobStatus
from app.pipeline import run_full_pipeline

logger = logging.getLogger(__name__)

# Lyrics files each output format produces next to a song.
FORMAT_EXTENSIONS = {"lrc": [".lrc"], "txt": [".txt"], "all": [".lrc", ".txt"]}


class BatchFinished(Exception):
    """Raised when cancelling a batch that has already ended."""


class AlongsideWriteError(Exception):
    """Writing a result file next to its source song failed."""


@dataclass
class BatchItem:
    path: str
    status: ItemStatus = ItemStatus.pending
    job_id: str | None = None
    error: str | None = None
    written_alongside: list[str] = field(default_factory=list)


@dataclass
class Batch:
    id: str
    created_at: str
    output_mode: str
    options: dict
    items: list[BatchItem]
    status: BatchStatus = BatchStatus.pending
    completed_at: str | None = None
    cancel_requested: bool = False
    delete_requested: bool = False


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def targets_exist(source: Path, format: str) -> bool:
    """Whether every lyrics file *format* produces already exists next to *source*."""
    return all(
        (source.parent / f"{source.stem}{ext}").exists()
        for ext in FORMAT_EXTENSIONS.get(format, [f".{format}"])
    )


def write_alongside(
    output_files: list[str], source: Path, skip_existing: bool
) -> tuple[list[str], bool]:
    """Copy result files next to *source* as '<song stem>.<ext>'.

    Existing files are left alone when *skip_existing* (checked right before
    writing, so a file that appeared while the batch ran is never
    overwritten). Each file is written to a temp file and renamed into place.
    Returns (written paths, whether every target already existed).
    """
    written: list[str] = []
    existed = 0
    for output in output_files:
        target = source.parent / f"{source.stem}{Path(output).suffix}"
        if skip_existing and target.exists():
            existed += 1
            continue
        tmp = target.parent / f".{target.name}.{uuid.uuid4().hex[:8]}.tmp"
        try:
            shutil.copyfile(output, tmp)
            os.replace(tmp, target)
        except OSError as e:
            tmp.unlink(missing_ok=True)
            raise AlongsideWriteError(f"could not write next to source: {e}") from e
        written.append(str(target))
    return written, bool(output_files) and existed == len(output_files)


class BatchManager:
    def __init__(self, job_manager: jm.JobManager, pipeline_fn: Callable = run_full_pipeline):
        self.job_manager = job_manager
        self.pipeline_fn = pipeline_fn
        self._batches: OrderedDict[str, Batch] = OrderedDict()
        self._lock = threading.RLock()

    # ── Public API ──────────────────────────────────────────────────────

    def create_batch(self, paths: list[Path], options: dict) -> Batch:
        """Create a batch for *paths* and start processing it in the background."""
        output_mode = config.BATCH_OUTPUT
        skip_existing = options.get("skip_existing", True)
        items = []
        for path in paths:
            item = BatchItem(path=str(path))
            if (
                output_mode == "alongside"
                and skip_existing
                and targets_exist(Path(path), options.get("format", "lrc"))
            ):
                item.status = ItemStatus.skipped
            items.append(item)

        batch = Batch(
            id=uuid.uuid4().hex[:12],
            created_at=_now(),
            output_mode=output_mode,
            options=dict(options),
            items=items,
        )
        with self._lock:
            self._batches[batch.id] = batch
        self.evict_old()
        threading.Thread(target=self._run, args=(batch,), daemon=True).start()
        return batch

    def get(self, batch_id: str) -> Batch | None:
        with self._lock:
            return self._batches.get(batch_id)

    def list_batches(self) -> list[Batch]:
        with self._lock:
            return list(reversed(self._batches.values()))

    def cancel(self, batch_id: str) -> Batch:
        """Stop a batch: pending songs are cancelled, a running song finishes."""
        with self._lock:
            batch = self._require(batch_id)
            if batch.completed_at:
                raise BatchFinished(batch_id)
            self._request_cancel(batch)
            return batch

    def delete(self, batch_id: str) -> str:
        """Cancel and remove a batch; 'cancelling' while a song still runs."""
        with self._lock:
            batch = self._require(batch_id)
            batch.delete_requested = True
            if not batch.completed_at:
                self._request_cancel(batch)
            if any(i.status == ItemStatus.processing for i in batch.items):
                return "cancelling"
            self._purge(batch)
            return "deleted"

    def zip_results(self, batch_id: str) -> bytes | None:
        """All finished result files as a ZIP ('<song>/<file>'), None if none yet."""
        with self._lock:
            items = list(self._require(batch_id).items)
        buffer = io.BytesIO()
        folders: dict[str, int] = {}
        count = 0
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for item in items:
                job = self.job_manager.get_job(item.job_id) if item.job_id else None
                if not job or job.status != JobStatus.completed or not job.result:
                    continue
                stem = Path(item.path).stem
                folders[stem] = folders.get(stem, 0) + 1
                folder = stem if folders[stem] == 1 else f"{stem} ({folders[stem]})"
                for file_path in job.result:
                    if os.path.exists(file_path):
                        zf.write(file_path, f"{folder}/{os.path.basename(file_path)}")
                        count += 1
        return buffer.getvalue() if count else None

    def evict_old(self) -> None:
        """Remove ended batches (and their jobs) older than JOB_TTL_SECONDS."""
        now = datetime.now(timezone.utc)
        with self._lock:
            expired = [
                b for b in self._batches.values()
                if b.completed_at
                and (now - datetime.fromisoformat(b.completed_at)).total_seconds()
                > jm.JOB_TTL_SECONDS
            ]
            for batch in expired:
                self._purge(batch)
                logger.info(f"Evicted old batch {batch.id}")

    # ── Internals ───────────────────────────────────────────────────────

    def _require(self, batch_id: str) -> Batch:
        batch = self._batches.get(batch_id)
        if batch is None:
            raise KeyError(batch_id)
        return batch

    def _request_cancel(self, batch: Batch) -> None:
        # Caller holds the lock.
        batch.cancel_requested = True
        for item in batch.items:
            if item.status == ItemStatus.pending:
                item.status = ItemStatus.cancelled
        if not any(i.status == ItemStatus.processing for i in batch.items):
            self._finalize(batch)

    def _finalize(self, batch: Batch) -> None:
        # Caller holds the lock. Idempotent.
        if batch.completed_at:
            return
        if batch.cancel_requested:
            batch.status = BatchStatus.cancelled
        elif batch.items and all(i.status == ItemStatus.failed for i in batch.items):
            batch.status = BatchStatus.failed
        else:
            batch.status = BatchStatus.completed
        batch.completed_at = _now()
        logger.info(f"Batch {batch.id} {batch.status.value}")

    def _purge(self, batch: Batch) -> None:
        # Caller holds the lock.
        for item in batch.items:
            if item.job_id:
                self.job_manager.delete_job(item.job_id, force=True)
        self._batches.pop(batch.id, None)

    def _next_item(self, batch: Batch) -> BatchItem | None:
        with self._lock:
            if batch.cancel_requested:
                return None
            for item in batch.items:
                if item.status == ItemStatus.pending:
                    item.status = ItemStatus.processing
                    batch.status = BatchStatus.processing
                    return item
            return None

    def _run(self, batch: Batch) -> None:
        options = dict(batch.options)
        skip_existing = options.pop("skip_existing", True)
        while (item := self._next_item(batch)) is not None:
            job = self.job_manager.create_job(batch_id=batch.id)
            with self._lock:
                item.job_id = job.id
            self.job_manager.run(
                job,
                self.pipeline_fn,
                input_path=item.path,
                output_dir=job.output_dir,
                **options,
            )
            status, error, written = ItemStatus.completed, None, []
            if job.status != JobStatus.completed:
                status, error = ItemStatus.failed, job.error
            elif batch.output_mode == "alongside" and job.result:
                try:
                    written, all_existed = write_alongside(
                        job.result, Path(item.path), skip_existing
                    )
                    if all_existed:
                        status = ItemStatus.skipped
                except AlongsideWriteError as e:
                    status, error = ItemStatus.failed, str(e)
            with self._lock:
                item.status, item.error, item.written_alongside = status, error, written

        with self._lock:
            for item in batch.items:
                if item.status == ItemStatus.pending:
                    item.status = ItemStatus.cancelled
            self._finalize(batch)
            if batch.delete_requested and batch.id in self._batches:
                self._purge(batch)
        self.evict_old()
