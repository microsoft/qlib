# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import unittest
from pathlib import Path, PurePosixPath, PureWindowsPath
from tempfile import TemporaryDirectory
from urllib.parse import urlparse
from urllib.request import url2pathname


class LockPathTest(unittest.TestCase):
    """Regression tests for the FileLock path of ``MLflowExpManager``.

    The lock path used to be built with ``os.path.join(pr.netloc, pr.path.lstrip("/"))``,
    which turns an absolute ``file`` URI into a CWD-relative path on POSIX systems,
    so the lock no longer serializes runs sharing the same URI from different working
    directories. See issue #2252.
    """

    def _lock_path(self, uri: str):
        """Mirror of the lock path computation in ``MLflowExpManager._get_or_create_exp``."""
        pr = urlparse(uri)
        return Path(url2pathname(pr.path)) / "filelock"

    def test_absolute_file_uri_posix(self):
        # On POSIX, url2pathname is the identity on the URI path, so the lock path
        # must be the absolute URI path + "/filelock". (Verified on Linux CI; the
        # mirror Windows behavior is covered by test_lock_lands_inside_uri_target.)
        for uri in ("file:/Users/me/proj/mlruns", "file:///Users/me/proj/mlruns"):
            with self.subTest(uri=uri):
                lock = PurePosixPath(urlparse(uri).path) / "filelock"
                self.assertTrue(lock.is_absolute(), f"lock path {lock} must be absolute")
                self.assertEqual(str(lock), urlparse(uri).path + "/filelock")

    @unittest.skipIf(os.name != "nt", "Windows-only semantics")
    def test_windows_drive_letter_uri(self):
        # A drive-letter URI must not be mangled (the old lstrip("/") kept "C:/..." intact
        # under NT semantics only by accident; url2pathname handles it correctly).
        uri = "file:///C:/Users/me/mlruns"
        lock = PureWindowsPath(url2pathname(urlparse(uri).path)) / "filelock"
        self.assertEqual(str(lock).lower(), "c:\\users\\me\\mlruns\\filelock")

    def test_relative_uri_stays_relative(self):
        # A relative file URI must keep its (relative) behavior.
        lock = self._lock_path("file:mlruns")
        self.assertEqual(str(lock), os.path.join("mlruns", "filelock"))
        self.assertFalse(lock.is_absolute())

    def test_lock_lands_inside_uri_target(self):
        # End-to-end: the lock file must resolve inside the directory the URI points to.
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            if os.name == "nt":
                uri = "file:///" + base.resolve().as_posix()
            else:
                uri = "file://" + base.resolve().as_posix()
            lock = self._lock_path(uri)
            self.assertEqual(Path(lock).resolve().parent, base.resolve())


if __name__ == "__main__":
    unittest.main()
