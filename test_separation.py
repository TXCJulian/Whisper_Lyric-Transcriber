"""Tests for app/separation.py helpers (no models are loaded)."""
import os
import subprocess
import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent


class ResampleTests(unittest.TestCase):
    def test_resample_changes_rate_and_keeps_pitch(self):
        from app.separation import resample

        t = torch.arange(48000) / 48000
        tone = torch.sin(2 * torch.pi * 1000 * t).repeat(2, 1)  # 1 s stereo 1 kHz at 48 kHz
        out = resample(tone, 48000, 44100)
        self.assertEqual(tuple(out.shape), (2, 44100))
        peak_hz = torch.fft.rfft(out[0]).abs().argmax().item()  # 1 s of audio -> bin == Hz
        self.assertAlmostEqual(peak_hz, 1000, delta=2)

    def test_resample_same_rate_is_identity(self):
        from app.separation import resample

        x = torch.randn(2, 1000)
        self.assertIs(resample(x, 44100, 44100), x)

    def test_separation_does_not_need_torchaudio(self):
        # torchaudio stopped publishing releases after 2.11; the app must not depend on it.
        env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
        result = subprocess.run(
            [sys.executable, "-c", "import sys, app.separation; print('torchaudio' in sys.modules)"],
            cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "False")


if __name__ == "__main__":
    unittest.main()
