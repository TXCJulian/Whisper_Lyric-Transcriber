"""Tests for JobManager: warnings list, blocking run, batch-owned jobs."""
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import app.job_manager as jm
from app.models import JobStatus


class JobManagerTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        patcher = patch.object(jm, "JOBS_BASE_DIR", tmp.name)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.manager = jm.JobManager(max_workers=1)

    def test_warnings_accumulate_and_last_is_warning(self):
        def task(progress_callback, warning_callback):
            warning_callback("a")
            warning_callback("b")
            return []

        job = self.manager.create_job()
        self.manager.run(job, task)
        self.assertEqual(job.warnings, ["a", "b"])
        self.assertEqual(job.warning, "b")

    def test_run_blocks_until_done(self):
        job = self.manager.create_job()
        self.manager.run(job, lambda progress_callback, warning_callback, value: [value], value="x")
        self.assertEqual(job.status, JobStatus.completed)
        self.assertEqual(job.result, ["x"])

    def test_failed_task_marks_failed(self):
        def task(progress_callback, warning_callback):
            raise RuntimeError("boom")

        job = self.manager.create_job()
        self.manager.run(job, task)
        self.assertEqual(job.status, JobStatus.failed)
        self.assertEqual(job.error, "boom")

    def test_batch_job_not_evicted(self):
        old = (datetime.now(timezone.utc) - timedelta(seconds=jm.JOB_TTL_SECONDS + 60)).isoformat()
        plain = self.manager.create_job()
        owned = self.manager.create_job(batch_id="b1")
        for job in (plain, owned):
            job.status = JobStatus.completed
            job.completed_at = old
        self.manager._evict_old_jobs()
        self.assertIsNone(self.manager.get_job(plain.id))
        self.assertIs(self.manager.get_job(owned.id), owned)

    def test_delete_batch_job_refused_unless_forced(self):
        job = self.manager.create_job(batch_id="b1")
        job.status = JobStatus.completed
        with self.assertRaises(jm.JobInBatch) as ctx:
            self.manager.delete_job(job.id)
        self.assertEqual(ctx.exception.batch_id, "b1")
        self.assertIs(self.manager.get_job(job.id), job)
        self.assertTrue(self.manager.delete_job(job.id, force=True))
        self.assertIsNone(self.manager.get_job(job.id))

    def test_api_delete_batch_job_returns_409(self):
        from fastapi.testclient import TestClient
        import app.main as main

        with patch.object(main, "job_manager", self.manager):
            job = self.manager.create_job(batch_id="b1")
            job.status = JobStatus.completed
            response = TestClient(main.app).delete(f"/jobs/{job.id}")
        self.assertEqual(response.status_code, 409)
        detail = response.json()["detail"]
        self.assertEqual(detail["code"], "job_in_batch")
        self.assertEqual(detail["batch_id"], "b1")


if __name__ == "__main__":
    unittest.main()
