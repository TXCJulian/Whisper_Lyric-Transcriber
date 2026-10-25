"""Settings from a .env file reach modules that read the environment at import."""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


class DotenvTests(unittest.TestCase):
    def test_env_file_in_working_directory_reaches_config(self):
        with tempfile.TemporaryDirectory() as cwd:
            Path(cwd, ".env").write_text("CORRECTION_MIN_MATCH=42\n", encoding="utf-8")
            env = {k: v for k, v in os.environ.items() if k != "CORRECTION_MIN_MATCH"}
            env["PYTHONPATH"] = str(REPO_ROOT)
            result = subprocess.run(
                [sys.executable, "-c", "import app.config as c; print(c.CORRECTION_MIN_MATCH)"],
                cwd=cwd, env=env, capture_output=True, text=True, timeout=60,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "42.0")


if __name__ == "__main__":
    unittest.main()
