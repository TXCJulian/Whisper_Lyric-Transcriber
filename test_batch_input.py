"""Tests for app/batch_input.py (directory listing and CSV parsing)."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import config
from app.batch_input import (
    BatchInputError,
    collect_batch_paths,
    list_directory_audio,
    parse_csv_paths,
)
from app.paths import PathNotAllowed


class BatchInputTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        base = Path(tmp.name).resolve()
        self.root = base / "music"
        self.album = self.root / "A"
        (self.album / "sub").mkdir(parents=True)
        for name in ("01 b.flac", "02 a.mp3", "cover.jpg", "sub/03.flac"):
            (self.album / name).write_bytes(b"x")
        self.outside = base / "outside.flac"
        self.outside.write_bytes(b"x")
        self.roots = [self.root]
        patcher = patch.object(config, "ALLOWED_ROOTS", self.roots)
        patcher.start()
        self.addCleanup(patcher.stop)

    def csv(self, *lines: str, newline: str = "\n") -> bytes:
        return (newline.join(lines) + newline).encode("utf-8")

    def test_directory_lists_audio_sorted_non_recursive(self):
        paths = list_directory_audio(str(self.album), recursive=False, roots=self.roots)
        self.assertEqual([p.name for p in paths], ["01 b.flac", "02 a.mp3"])

    def test_directory_recursive_includes_subfolders(self):
        paths = list_directory_audio(str(self.album), recursive=True, roots=self.roots)
        self.assertEqual(
            [p.relative_to(self.album).as_posix() for p in paths],
            ["01 b.flac", "02 a.mp3", "sub/03.flac"],
        )

    def test_directory_without_audio_rejected(self):
        empty = self.root / "empty"
        empty.mkdir()
        (empty / "notes.txt").write_bytes(b"x")
        with self.assertRaises(BatchInputError) as ctx:
            list_directory_audio(str(empty), recursive=False, roots=self.roots)
        self.assertEqual(ctx.exception.code, "invalid_batch_request")

    def test_directory_outside_root(self):
        with self.assertRaises(PathNotAllowed):
            list_directory_audio(str(self.outside.parent), recursive=False, roots=self.roots)

    def test_csv_header_comments_blank_and_dupes(self):
        p1, p2 = self.album / "01 b.flac", self.album / "02 a.mp3"
        content = self.csv("path", "# a comment", "", str(p1), str(p1), str(p2))
        self.assertEqual(parse_csv_paths(content, self.roots), [p1, p2])

    def test_csv_quoted_path_with_comma(self):
        song = self.album / "01-13 Illegal, legal, egal.flac"
        song.write_bytes(b"x")
        content = self.csv("path", f'"{song}"')
        self.assertEqual(parse_csv_paths(content, self.roots), [song])

    def test_csv_unquoted_comma_reports_line(self):
        song = self.album / "Illegal, legal.flac"
        song.write_bytes(b"x")
        with self.assertRaises(BatchInputError) as ctx:
            parse_csv_paths(self.csv("path", str(song)), self.roots)
        self.assertEqual(ctx.exception.code, "invalid_csv")
        self.assertEqual(ctx.exception.errors[0]["line"], 2)
        self.assertIn("extra columns", ctx.exception.errors[0]["reason"])

    def test_csv_unicode_path(self):
        song = self.album / "Café – Ünïcode.flac"
        song.write_bytes(b"x")
        self.assertEqual(parse_csv_paths(self.csv(str(song)), self.roots), [song])

    def test_csv_crlf_and_bom(self):
        p1 = self.album / "01 b.flac"
        content = b"\xef\xbb\xbf" + self.csv("path", str(p1), newline="\r\n")
        self.assertEqual(parse_csv_paths(content, self.roots), [p1])

    def test_csv_collects_all_errors(self):
        content = self.csv(
            str(self.album / "01 b.flac"),
            str(self.album / "cover.jpg"),
            str(self.album / "missing.flac"),
            str(self.outside),
        )
        with self.assertRaises(BatchInputError) as ctx:
            parse_csv_paths(content, self.roots)
        self.assertEqual([e["line"] for e in ctx.exception.errors], [2, 3, 4])

    def test_collect_requires_exactly_one_source(self):
        for directory, csv_content in ((None, None), (str(self.album), b"x")):
            with self.subTest(directory=directory), self.assertRaises(BatchInputError) as ctx:
                collect_batch_paths(directory, csv_content, recursive=False)
            self.assertEqual(ctx.exception.code, "invalid_batch_request")

    def test_collect_enforces_max_size(self):
        with patch.object(config, "MAX_BATCH_SIZE", 2), self.assertRaises(BatchInputError) as ctx:
            collect_batch_paths(str(self.album), None, recursive=True)
        self.assertEqual(ctx.exception.code, "invalid_batch_request")
        self.assertIn("2", ctx.exception.message)


if __name__ == "__main__":
    unittest.main()
