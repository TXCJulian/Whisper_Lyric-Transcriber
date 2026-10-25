"""Tests for the Genius correction threshold (CORRECTION_MIN_MATCH)."""
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from app.correction import correct_transcription
from app.transcription import Segment, WordTiming


def make_segments(text: str) -> list[Segment]:
    words = [WordTiming(start=i, end=i + 0.9, word=w) for i, w in enumerate(text.split())]
    return [Segment(start=0, end=len(words), text=text, words=words)]


def words_of(segments: list[Segment]) -> list[str]:
    return [w.word for seg in segments for w in seg.words]


class CorrectionThresholdTests(unittest.TestCase):
    def test_zero_threshold_keeps_todays_behaviour(self):
        result = correct_transcription(make_segments("hello wrld foo"), "hello world foo", min_match=0)
        self.assertIn("world", words_of(result))

    def test_below_threshold_returns_original_and_warns(self):
        original = make_segments("hello wrld foo")
        warnings: list[str] = []
        result = correct_transcription(
            original,
            "completely different lyrics here",
            min_match=30,
            warning_callback=warnings.append,
        )
        self.assertEqual(words_of(result), ["hello", "wrld", "foo"])
        self.assertEqual(len(warnings), 1)
        self.assertTrue(warnings[0].startswith("Genius lyrics matched only"))
        self.assertIn("threshold 30 %", warnings[0])

    def test_at_or_above_threshold_corrects(self):
        warnings: list[str] = []
        result = correct_transcription(
            make_segments("Hello world foo"), "hello world foo",
            min_match=100, warning_callback=warnings.append,
        )
        self.assertEqual(words_of(result), ["hello", "world", "foo"])
        self.assertEqual(warnings, [])

    def test_pipeline_forwards_threshold(self):
        import app.pipeline as pipeline

        engine = MagicMock()
        engine.transcribe.return_value = (make_segments("hello world"), "en")
        def on_warning(message):
            pass

        fake_correct = MagicMock(side_effect=lambda segments, lyrics, **kw: segments)
        with tempfile.TemporaryDirectory() as out, \
                patch.object(pipeline, "get_engine", return_value=engine), \
                patch.object(pipeline, "get_metadata_from_file", return_value=(None, None)), \
                patch.object(pipeline, "fetch_genius_lyrics", return_value="hello world"), \
                patch.object(pipeline, "correct_transcription", fake_correct):
            pipeline.run_full_pipeline(
                input_path=f"{out}/song.flac", output_dir=out, no_separation=True,
                artist="a", title="t", correction_min_match=40,
                warning_callback=on_warning,
            )
        kwargs = fake_correct.call_args.kwargs
        self.assertEqual(kwargs["min_match"], 40)
        self.assertIs(kwargs["warning_callback"], on_warning)

    def test_routes_validate_threshold_range(self):
        from fastapi.testclient import TestClient
        import app.job_manager as jm
        import app.main as main

        client = TestClient(main.app)
        with tempfile.TemporaryDirectory() as jobs_dir,                 patch.object(jm, "JOBS_BASE_DIR", jobs_dir),                 patch.object(main, "job_manager", jm.JobManager()),                 patch.object(main.job_manager, "submit") as submit:
            for path, extra in (("/transcribe", {}), ("/correct", {"artist": "a", "title": "t"})):
                with self.subTest(path=path):
                    bad = client.post(path, files={"file": ("s.flac", b"x")},
                                      data={"correction_min_match": "150", **extra})
                    self.assertEqual(bad.status_code, 422)
                    ok = client.post(path, files={"file": ("s.flac", b"x")},
                                     data={"correction_min_match": "35", **extra})
                    self.assertEqual(ok.status_code, 200)
                    self.assertEqual(submit.call_args.kwargs["correction_min_match"], 35)


if __name__ == "__main__":
    unittest.main()
