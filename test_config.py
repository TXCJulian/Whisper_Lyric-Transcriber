"""Tests for app/config.py env parsing; run with python -m unittest discover."""
import unittest
from pathlib import Path
from unittest.mock import patch

from app import config


class ConfigParsingTests(unittest.TestCase):
    def test_parse_roots_splits_strips_dedupes(self):
        roots = config.parse_roots("/music, /music ,,/other")
        self.assertEqual(roots, [Path("/music").resolve(), Path("/other").resolve()])

    def test_parse_roots_empty(self):
        self.assertEqual(config.parse_roots(""), [])
        self.assertEqual(config.parse_roots("  "), [])

    def test_batch_output_valid_values(self):
        self.assertEqual(config.parse_batch_output("download"), "download")
        self.assertEqual(config.parse_batch_output("ALONGSIDE "), "alongside")

    def test_batch_output_invalid_falls_back(self):
        with self.assertLogs("app.config", level="ERROR"):
            self.assertEqual(config.parse_batch_output("elsewhere"), "download")

    def test_min_match_parsing(self):
        self.assertEqual(config.parse_min_match("30"), 30.0)
        self.assertEqual(config.parse_min_match("12.5"), 12.5)
        self.assertEqual(config.parse_min_match(""), 0.0)

    def test_min_match_invalid_falls_back(self):
        for raw in ("abc", "-1", "101"):
            with self.subTest(raw=raw), self.assertLogs("app.config", level="ERROR"):
                self.assertEqual(config.parse_min_match(raw), 0.0)

    def test_resolve_min_match_precedence(self):
        with patch.object(config, "CORRECTION_MIN_MATCH", 25.0):
            self.assertEqual(config.resolve_min_match(None), 25.0)
            self.assertEqual(config.resolve_min_match(10), 10.0)
            with self.assertRaises(ValueError):
                config.resolve_min_match(101)

    def test_audio_extensions(self):
        self.assertIn(".flac", config.AUDIO_EXTENSIONS)
        self.assertNotIn(".txt", config.AUDIO_EXTENSIONS)
        self.assertTrue(all(ext == ext.lower() and ext.startswith(".") for ext in config.AUDIO_EXTENSIONS))


if __name__ == "__main__":
    unittest.main()
