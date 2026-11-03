"""Tests for app/pipeline.py housekeeping (no GPU, no models)."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import app.pipeline as pipeline


class PipelineCleanupTests(unittest.TestCase):
    def test_failed_transcription_removes_vocals_file(self):
        # A failing song must not leave its separated vocals (tens of MB) behind:
        # batch jobs live until their batch expires.
        with tempfile.TemporaryDirectory() as out:
            vocals = Path(out) / "song_vocals.wav"

            def fake_separate(input_path, output_dir, model_name):
                vocals.write_bytes(b"x" * 1024)
                return str(vocals)

            engine = MagicMock()
            engine.transcribe.side_effect = RuntimeError("CUDA out of memory")
            with patch.object(pipeline, "separate_vocals", fake_separate), \
                    patch.object(pipeline, "unload_separation_model"), \
                    patch.object(pipeline, "get_metadata_from_file", return_value=(None, None)), \
                    patch.object(pipeline, "get_engine", return_value=engine):
                with self.assertRaises(RuntimeError):
                    pipeline.run_full_pipeline(input_path=f"{out}/song.flac", output_dir=out)
            self.assertFalse(vocals.exists())

    def _run_with_failing_cleanup(self, out: str, transcribe_error: Exception | None):
        vocals = Path(out) / "song_vocals.wav"

        def fake_separate(input_path, output_dir, model_name):
            vocals.write_bytes(b"x")
            return str(vocals)

        engine = MagicMock()
        if transcribe_error:
            engine.transcribe.side_effect = transcribe_error
        else:
            engine.transcribe.return_value = ([], "en")  # no vocals -> no output files
        with patch.object(pipeline, "separate_vocals", fake_separate), \
                patch.object(pipeline, "unload_separation_model"), \
                patch.object(pipeline, "get_metadata_from_file", return_value=(None, None)), \
                patch.object(pipeline, "get_engine", return_value=engine), \
                patch.object(pipeline.os, "remove", side_effect=PermissionError("busy")):
            return pipeline.run_full_pipeline(input_path=f"{out}/song.flac", output_dir=out)

    def test_cleanup_error_does_not_fail_successful_run(self):
        with tempfile.TemporaryDirectory() as out:
            self.assertEqual(self._run_with_failing_cleanup(out, None), [])

    def test_cleanup_error_does_not_mask_original_error(self):
        with tempfile.TemporaryDirectory() as out:
            with self.assertRaisesRegex(RuntimeError, "CUDA out of memory"):
                self._run_with_failing_cleanup(out, RuntimeError("CUDA out of memory"))


if __name__ == "__main__":
    unittest.main()
