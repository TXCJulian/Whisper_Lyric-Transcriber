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


if __name__ == "__main__":
    unittest.main()
