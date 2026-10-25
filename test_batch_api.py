"""Tests for the batch HTTP endpoints (app/batch_routes.py)."""
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import app.batch_routes as batch_routes
import app.job_manager as jm
import app.main as main
from app import config
from app.batch_manager import BatchManager
from test_batch_manager import FakePipeline


class BatchApiTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        base = Path(tmp.name).resolve()
        self.root = base / "music"
        self.album = self.root / "Album"
        self.album.mkdir(parents=True)
        self.songs = []
        for name in ("01 a.flac", "02 b.flac"):
            path = self.album / name
            path.write_bytes(b"audio")
            self.songs.append(path)

        patch.object(jm, "JOBS_BASE_DIR", str(base / "jobs")).start()
        self.jobs = jm.JobManager(max_workers=1)
        self.batches = BatchManager(self.jobs, pipeline_fn=FakePipeline())
        for patcher in (
            patch.object(config, "ALLOWED_ROOTS", [self.root]),
            patch.object(config, "BATCH_OUTPUT", "download"),
            patch.object(main, "job_manager", self.jobs),
            patch.object(batch_routes, "batch_manager", self.batches),
        ):
            patcher.start()
        self.addCleanup(patch.stopall)
        self.client = TestClient(main.app)

    def start_batch(self, **data):
        response = self.client.post("/batch", data=data)
        self.assertEqual(response.status_code, 202, response.text)
        return response.json()

    def wait_done(self, batch_id: str) -> dict:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            body = self.client.get(f"/batches/{batch_id}").json()
            if body["completed_at"]:
                return body
            time.sleep(0.02)
        self.fail("batch did not finish")

    def test_batch_disabled_without_roots(self):
        with patch.object(config, "ALLOWED_ROOTS", []):
            response = self.client.post("/batch", data={"directory": str(self.album)})
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json()["detail"]["code"], "batch_disabled")

    def test_batch_from_directory(self):
        created = self.start_batch(directory=str(self.album))
        self.assertEqual(created["total"], 2)
        body = self.wait_done(created["batch_id"])
        self.assertEqual(body["status"], "completed")
        self.assertEqual(body["counts"]["completed"], 2)
        self.assertEqual([i["path"] for i in body["items"]], [str(p) for p in self.songs])
        self.assertEqual(body["items"][0]["outputs"], ["01 a.lrc"])

    def test_batch_from_csv_upload(self):
        csv = ("path\n" + "\n".join(str(p) for p in self.songs) + "\n").encode()
        response = self.client.post("/batch", files={"csv": ("list.csv", csv, "text/csv")})
        self.assertEqual(response.status_code, 202, response.text)
        self.assertEqual(response.json()["total"], 2)

    def test_neither_or_both_sources_422(self):
        for kwargs in ({}, {"data": {"directory": str(self.album)},
                            "files": {"csv": ("l.csv", b"x", "text/csv")}}):
            with self.subTest(kwargs=list(kwargs)):
                response = self.client.post("/batch", **kwargs)
                self.assertEqual(response.status_code, 422)
                self.assertEqual(response.json()["detail"]["code"], "invalid_batch_request")

    def test_invalid_csv_lists_lines(self):
        csv = f"path\n{self.album / 'missing.flac'}\n".encode()
        response = self.client.post("/batch", files={"csv": ("l.csv", csv, "text/csv")})
        self.assertEqual(response.status_code, 422)
        detail = response.json()["detail"]
        self.assertEqual(detail["code"], "invalid_csv")
        self.assertEqual(detail["errors"][0]["line"], 2)

    def test_path_outside_root_403(self):
        response = self.client.post("/batch", data={"directory": "/etc"})
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json()["detail"]["code"], "path_not_allowed")

    def test_threshold_out_of_range_422(self):
        response = self.client.post(
            "/batch", data={"directory": str(self.album), "correction_min_match": "150"}
        )
        self.assertEqual(response.status_code, 422)

    def test_list_get_result_cancel_delete(self):
        batch_id = self.start_batch(directory=str(self.album))["batch_id"]
        self.wait_done(batch_id)
        listed = self.client.get("/batches").json()
        self.assertIn(batch_id, [b["batch_id"] for b in listed])

        result = self.client.get(f"/batches/{batch_id}/result")
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.headers["content-type"], "application/zip")

        cancel = self.client.post(f"/batches/{batch_id}/cancel")
        self.assertEqual(cancel.status_code, 409)
        self.assertEqual(cancel.json()["detail"]["code"], "batch_finished")

        deleted = self.client.delete(f"/batches/{batch_id}")
        self.assertEqual(deleted.json(), {"status": "deleted", "batch_id": batch_id})
        missing = self.client.get(f"/batches/{batch_id}")
        self.assertEqual(missing.status_code, 404)
        self.assertEqual(missing.json()["detail"]["code"], "batch_not_found")

    def test_item_job_readable_via_jobs_endpoint(self):
        body = self.wait_done(self.start_batch(directory=str(self.album))["batch_id"])
        job = self.client.get(f"/jobs/{body['items'][0]['job_id']}")
        self.assertEqual(job.status_code, 200)
        self.assertEqual(job.json()["status"], "completed")

    def test_health_reports_batch_and_threshold(self):
        health = self.client.get("/health").json()
        self.assertEqual(
            health["batch"],
            {"enabled": True, "allowed_roots": [str(self.root)], "output_mode": "download"},
        )
        self.assertEqual(health["correction_min_match"], config.CORRECTION_MIN_MATCH)


if __name__ == "__main__":
    unittest.main()
