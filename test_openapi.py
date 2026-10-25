"""The OpenAPI schema (/docs, /redoc) documents every endpoint and field."""
import unittest

from fastapi.testclient import TestClient

import app.main as main

HTTP_METHODS = {"get", "post", "put", "patch", "delete"}


class OpenApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        main.app.openapi_schema = None  # rebuild in case another test cached it
        cls.schema = main.app.openapi()

    def operations(self):
        for path, item in self.schema["paths"].items():
            for method, op in item.items():
                if method in HTTP_METHODS:
                    yield path, method, op

    def resolve(self, schema: dict) -> dict:
        ref = schema.get("$ref")
        if ref:
            return self.schema["components"]["schemas"][ref.rsplit("/", 1)[-1]]
        return schema

    def test_every_operation_has_summary_description_and_tag(self):
        for path, method, op in self.operations():
            with self.subTest(op=f"{method.upper()} {path}"):
                self.assertTrue(op.get("summary"), "missing summary")
                self.assertTrue(op.get("description"), "missing description")
                self.assertTrue(op.get("tags"), "missing tag")

    def test_every_form_field_has_description(self):
        for path, method, op in self.operations():
            content = op.get("requestBody", {}).get("content", {})
            form = content.get("multipart/form-data") or content.get("application/x-www-form-urlencoded")
            if not form:
                continue
            for name, prop in self.resolve(form["schema"]).get("properties", {}).items():
                with self.subTest(op=f"{method.upper()} {path}", field=name):
                    self.assertTrue(prop.get("description"), "missing description")

    def test_error_statuses_documented(self):
        expected = {
            ("get", "/jobs/{job_id}"): {"404"},
            ("get", "/jobs/{job_id}/result"): {"204", "404", "409"},
            ("delete", "/jobs/{job_id}"): {"404", "409"},
            ("post", "/batch"): {"202", "403", "404", "422"},
            ("delete", "/batches/{batch_id}"): {"200", "202", "404"},
        }
        for (method, path), codes in expected.items():
            with self.subTest(op=f"{method.upper()} {path}"):
                documented = set(self.schema["paths"][path][method]["responses"])
                self.assertTrue(codes <= documented, f"missing {codes - documented}")

    def test_result_endpoint_documents_both_media_types(self):
        content = self.schema["paths"]["/jobs/{job_id}/result"]["get"]["responses"]["200"]["content"]
        self.assertTrue({"application/octet-stream", "application/zip"} <= set(content))

    def test_api_description_mentions_batch_env(self):
        self.assertIn("TRANSCRIBER_ALLOWED_ROOTS", self.schema["info"]["description"])

    def test_health_json_fields_unchanged(self):
        health = TestClient(main.app).get("/health").json()
        expected = {
            "status", "version", "runtime_profile", "gpu_backend", "gpu_name",
            "transcription_engine", "vram_total_mb", "whisper_model_fit",
            "batch", "correction_min_match",
        }
        self.assertTrue(expected <= set(health), f"missing {expected - set(health)}")


if __name__ == "__main__":
    unittest.main()
