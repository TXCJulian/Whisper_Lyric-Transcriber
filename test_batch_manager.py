"""Tests for app/batch_manager.py with a fake pipeline (no GPU, no models)."""
import io
import tempfile
import threading
import time
import unittest
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import app.batch_manager as bm
import app.job_manager as jm
from app import config
from app.models import BatchStatus, ItemStatus


class FakePipeline:
    """Writes '<stem>.<ext>' into output_dir; fails for names containing FAIL."""

    def __init__(self):
        self.calls: list[str] = []
        self.first_started = threading.Event()
        self.release_first = threading.Event()
        self.release_first.set()  # no blocking unless a test clears it

    def __call__(self, input_path, output_dir, format="lrc",
                 progress_callback=None, warning_callback=None, **_):
        self.calls.append(input_path)
        if len(self.calls) == 1:
            self.first_started.set()
            self.release_first.wait(5)
        if "FAIL" in Path(input_path).name:
            raise RuntimeError("fake failure")
        exts = [".lrc", ".txt"] if format == "all" else [f".{format}"]
        outputs = []
        for ext in exts:
            out = Path(output_dir) / f"{Path(input_path).stem}{ext}"
            out.write_text(f"generated {out.name}", encoding="utf-8")
            outputs.append(str(out))
        return outputs


class BatchManagerTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        base = Path(tmp.name)
        self.music = base / "music"
        self.music.mkdir()
        for patcher in (
            patch.object(jm, "JOBS_BASE_DIR", str(base / "jobs")),
            patch.object(config, "BATCH_OUTPUT", "download"),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.jobs = jm.JobManager(max_workers=1)
        self.pipeline = FakePipeline()
        self.manager = bm.BatchManager(self.jobs, pipeline_fn=self.pipeline)

    def songs(self, *names: str) -> list[Path]:
        paths = []
        for name in names:
            path = self.music / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"audio")
            paths.append(path)
        return paths

    def start(self, paths, **options):
        options.setdefault("format", "lrc")
        options.setdefault("skip_existing", True)
        return self.manager.create_batch(paths, options)

    def wait_ended(self, batch_id: str):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            batch = self.manager.get(batch_id)
            if batch is None or batch.completed_at:
                return batch
            time.sleep(0.02)
        self.fail("batch did not end in time")

    def statuses(self, batch):
        return [item.status for item in batch.items]

    def test_items_run_in_order_and_batch_completes(self):
        paths = self.songs("01.flac", "02.flac", "03.flac")
        batch = self.wait_ended(self.start(paths).id)
        self.assertEqual(self.pipeline.calls, [str(p) for p in paths])
        self.assertEqual(batch.status, BatchStatus.completed)
        self.assertEqual(self.statuses(batch), [ItemStatus.completed] * 3)

    def test_failed_item_does_not_stop_batch(self):
        batch = self.wait_ended(self.start(self.songs("01.flac", "02 FAIL.flac", "03.flac")).id)
        self.assertEqual(
            self.statuses(batch),
            [ItemStatus.completed, ItemStatus.failed, ItemStatus.completed],
        )
        self.assertEqual(batch.items[1].error, "fake failure")
        self.assertEqual(batch.status, BatchStatus.completed)

    def test_all_failed_marks_batch_failed(self):
        batch = self.wait_ended(self.start(self.songs("a FAIL.flac", "b FAIL.flac")).id)
        self.assertEqual(batch.status, BatchStatus.failed)

    def test_cancel_lets_running_item_finish_and_skips_pending(self):
        self.pipeline.release_first.clear()
        batch = self.start(self.songs("01.flac", "02.flac", "03.flac"))
        self.assertTrue(self.pipeline.first_started.wait(5))
        self.manager.cancel(batch.id)
        self.pipeline.release_first.set()
        batch = self.wait_ended(batch.id)
        self.assertEqual(
            self.statuses(batch),
            [ItemStatus.completed, ItemStatus.cancelled, ItemStatus.cancelled],
        )
        self.assertEqual(batch.status, BatchStatus.cancelled)
        self.assertEqual(len(self.pipeline.calls), 1)

    def test_cancel_finished_batch_raises(self):
        batch = self.wait_ended(self.start(self.songs("01.flac")).id)
        with self.assertRaises(bm.BatchFinished):
            self.manager.cancel(batch.id)

    def test_delete_while_running_returns_cancelling_then_purges(self):
        self.pipeline.release_first.clear()
        batch = self.start(self.songs("01.flac", "02.flac"))
        self.assertTrue(self.pipeline.first_started.wait(5))
        job_id = batch.items[0].job_id
        self.assertEqual(self.manager.delete(batch.id), "cancelling")
        self.pipeline.release_first.set()
        self.assertIsNone(self.wait_ended(batch.id))
        self.assertIsNone(self.jobs.get_job(job_id))

    def test_alongside_writes_next_to_source(self):
        (song,) = self.songs("Album/01 song.flac")
        with patch.object(config, "BATCH_OUTPUT", "alongside"):
            batch = self.wait_ended(self.start([song]).id)
        target = song.with_suffix(".lrc")
        self.assertEqual(target.read_text(encoding="utf-8"), "generated 01 song.lrc")
        self.assertEqual(song.read_bytes(), b"audio")
        self.assertEqual(batch.items[0].written_alongside, [str(target)])

    def test_alongside_skip_existing_at_creation(self):
        (song,) = self.songs("01.flac")
        song.with_suffix(".lrc").write_text("mine", encoding="utf-8")
        with patch.object(config, "BATCH_OUTPUT", "alongside"):
            batch = self.wait_ended(self.start([song]).id)
        self.assertEqual(self.statuses(batch), [ItemStatus.skipped])
        self.assertEqual(self.pipeline.calls, [])
        self.assertEqual(song.with_suffix(".lrc").read_text(encoding="utf-8"), "mine")

    def test_alongside_existing_file_appearing_midway_kept(self):
        first, second = self.songs("01.flac", "02.flac")
        self.pipeline.release_first.clear()
        with patch.object(config, "BATCH_OUTPUT", "alongside"):
            batch = self.start([first, second])
        self.assertTrue(self.pipeline.first_started.wait(5))
        second.with_suffix(".lrc").write_text("appeared", encoding="utf-8")
        self.pipeline.release_first.set()
        batch = self.wait_ended(batch.id)
        self.assertEqual(second.with_suffix(".lrc").read_text(encoding="utf-8"), "appeared")
        self.assertEqual(self.statuses(batch), [ItemStatus.completed, ItemStatus.skipped])

    def test_alongside_format_all_writes_only_missing(self):
        (song,) = self.songs("01.flac")
        song.with_suffix(".lrc").write_text("mine", encoding="utf-8")
        with patch.object(config, "BATCH_OUTPUT", "alongside"):
            batch = self.wait_ended(self.start([song], format="all").id)
        self.assertEqual(song.with_suffix(".lrc").read_text(encoding="utf-8"), "mine")
        self.assertEqual(song.with_suffix(".txt").read_text(encoding="utf-8"), "generated 01.txt")
        self.assertEqual(self.statuses(batch), [ItemStatus.completed])

    def test_alongside_readonly_target_fails_item(self):
        (song,) = self.songs("01.flac")
        with patch.object(config, "BATCH_OUTPUT", "alongside"), \
                patch.object(bm.os, "replace", side_effect=PermissionError("read-only")):
            batch = self.wait_ended(self.start([song]).id)
        item = batch.items[0]
        self.assertEqual(item.status, ItemStatus.failed)
        self.assertTrue(item.error.startswith("could not write next to source"))
        self.assertTrue(all(Path(f).exists() for f in self.jobs.get_job(item.job_id).result))
        self.assertFalse(song.with_suffix(".lrc").exists())

    def test_alongside_readonly_filesystem_fails_item_and_batch_ends(self):
        # On a read-only mount even creating (and cleaning up) the temp file fails;
        # the cleanup error must not escape and kill the runner.
        (song,) = self.songs("01.flac")
        erofs = OSError(30, "Read-only file system")
        with patch.object(config, "BATCH_OUTPUT", "alongside"), \
                patch.object(bm.shutil, "copyfile", side_effect=erofs), \
                patch.object(bm.Path, "unlink", side_effect=erofs):
            batch = self.wait_ended(self.start([song]).id)
        self.assertEqual(batch.items[0].status, ItemStatus.failed)
        self.assertIn("could not write next to source", batch.items[0].error)

    def test_unexpected_error_after_song_fails_item_not_runner(self):
        first, second = self.songs("01.flac", "02.flac")
        with patch.object(config, "BATCH_OUTPUT", "alongside"), \
                patch.object(bm, "write_alongside", side_effect=[RuntimeError("bug"), ([], False)]):
            batch = self.wait_ended(self.start([first, second]).id)
        self.assertEqual(self.statuses(batch), [ItemStatus.failed, ItemStatus.completed])
        self.assertIn("bug", batch.items[0].error)
        self.assertEqual(batch.status, BatchStatus.completed)

    def test_download_mode_never_writes_next_to_source(self):
        (song,) = self.songs("01.flac")
        self.wait_ended(self.start([song]).id)
        self.assertEqual(sorted(p.name for p in self.music.iterdir()), ["01.flac"])

    def test_zip_disambiguates_duplicate_stems(self):
        batch = self.wait_ended(self.start(self.songs("A/x.flac", "B/x.flac")).id)
        with zipfile.ZipFile(io.BytesIO(self.manager.zip_results(batch.id))) as z:
            self.assertEqual(set(z.namelist()), {"x/x.lrc", "x (2)/x.lrc"})

    def test_listing_evicts_expired_batches(self):
        # A finished batch must expire even if no other batch is ever started.
        batch = self.wait_ended(self.start(self.songs("01.flac")).id)
        job_id = batch.items[0].job_id
        batch.completed_at = (
            datetime.now(timezone.utc) - timedelta(seconds=jm.JOB_TTL_SECONDS + 60)
        ).isoformat()
        self.assertEqual(self.manager.list_batches(), [])
        self.assertIsNone(self.manager.get(batch.id))
        self.assertIsNone(self.jobs.get_job(job_id))

    def test_eviction_removes_batch_and_jobs(self):
        batch = self.wait_ended(self.start(self.songs("01.flac")).id)
        job_id = batch.items[0].job_id
        batch.completed_at = (
            datetime.now(timezone.utc) - timedelta(seconds=jm.JOB_TTL_SECONDS + 60)
        ).isoformat()
        self.manager.evict_old()
        self.assertIsNone(self.manager.get(batch.id))
        self.assertIsNone(self.jobs.get_job(job_id))


if __name__ == "__main__":
    unittest.main()
