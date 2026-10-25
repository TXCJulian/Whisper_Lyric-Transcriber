"""Tests for app/paths.py (TRANSCRIBER_ALLOWED_ROOTS checks)."""
import os
import tempfile
import unittest
from pathlib import Path

from app.paths import BatchDisabled, PathNotAllowed, PathNotFound, resolve_allowed_path


class ResolveAllowedPathTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.base = Path(tmp.name).resolve()
        self.root = self.base / "music"
        self.root.mkdir()
        self.song = self.root / "song.flac"
        self.song.write_bytes(b"x")
        self.outside = self.base / "outside"
        self.outside.mkdir()
        (self.outside / "secret.flac").write_bytes(b"x")

    def symlink(self, link: Path, target: Path):
        try:
            os.symlink(target, link, target_is_directory=target.is_dir())
        except (OSError, NotImplementedError) as e:
            self.skipTest(f"symlinks not available: {e}")

    def test_file_inside_root_resolves(self):
        self.assertEqual(resolve_allowed_path(str(self.song), [self.root]), self.song)

    def test_root_itself_allowed(self):
        self.assertEqual(resolve_allowed_path(str(self.root), [self.root]), self.root)

    def test_dotdot_escape_rejected(self):
        with self.assertRaises(PathNotAllowed):
            resolve_allowed_path(str(self.root / ".." / "outside" / "secret.flac"), [self.root])

    def test_symlink_escape_rejected(self):
        link = self.root / "link"
        self.symlink(link, self.outside)
        with self.assertRaises(PathNotAllowed):
            resolve_allowed_path(str(link / "secret.flac"), [self.root])

    def test_symlinked_root_is_resolved(self):
        link_root = self.base / "music-link"
        self.symlink(link_root, self.root)
        resolved = resolve_allowed_path(str(link_root / "song.flac"), [link_root])
        self.assertEqual(resolved, self.song)

    def test_sibling_prefix_rejected(self):
        sibling = self.base / "music2"
        sibling.mkdir()
        (sibling / "a.flac").write_bytes(b"x")
        with self.assertRaises(PathNotAllowed):
            resolve_allowed_path(str(sibling / "a.flac"), [self.root])

    def test_missing_path(self):
        with self.assertRaises(PathNotFound) as ctx:
            resolve_allowed_path(str(self.root / "missing.flac"), [self.root])
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.code, "path_not_found")

    def test_relative_path_rejected(self):
        with self.assertRaises(PathNotAllowed):
            resolve_allowed_path("music/song.flac", [self.root])

    def test_no_roots_configured(self):
        with self.assertRaises(BatchDisabled) as ctx:
            resolve_allowed_path(str(self.song), [])
        self.assertEqual(ctx.exception.status_code, 403)
        self.assertEqual(ctx.exception.code, "batch_disabled")


if __name__ == "__main__":
    unittest.main()
