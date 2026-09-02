# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import contextlib
import io
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from qlib.workflow.recorder import MLflowRecorder


class LogUncommittedCodeTest(unittest.TestCase):
    """Regression tests for ``MLflowRecorder._log_uncommitted_code``.

    The function used to run git with ``shell=True`` without capturing stderr, so in a
    non-git CWD every ``R.start()`` leaked a multi-line git usage banner to the caller's
    stderr and logged three misleading INFO records. See issue #2252 (secondary defect).
    """

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def _make_recorder(self):
        rec = MLflowRecorder.__new__(MLflowRecorder)  # skip __init__ (needs mlflow client)
        rec.client = mock.MagicMock()
        rec.id = "test-id"
        return rec

    def test_non_git_cwd_is_silent(self):
        """In a non-git CWD: no stderr leak, no INFO records, no exception."""
        rec = self._make_recorder()
        with contextlib.chdir(Path(self._tmp.name)):  # fresh dir, no .git
            err_buf = io.StringIO()
            with mock.patch("qlib.workflow.recorder.logger") as mock_logger:
                with contextlib.redirect_stderr(err_buf):
                    rec._log_uncommitted_code()
            self.assertNotIn("usage:", err_buf.getvalue())
            self.assertNotIn("fatal:", err_buf.getvalue().lower())
            mock_logger.info.assert_not_called()  # skip is DEBUG-level, not INFO

    def test_git_cwd_still_logs(self):
        """In a real git work tree the three diff artifacts are still produced."""
        repo = Path(__file__).resolve().parents[2]
        self.assertTrue((repo / ".git").exists(), "test must run inside the qlib git checkout")
        cwd = os.getcwd()
        os.chdir(repo)
        self.addCleanup(os.chdir, cwd)
        rec = self._make_recorder()
        rec._log_uncommitted_code()
        names = [c.args[2] for c in rec.client.log_text.call_args_list]
        self.assertIn("code_diff.txt", names)
        self.assertIn("code_status.txt", names)
        self.assertIn("code_cached.txt", names)


if __name__ == "__main__":
    unittest.main()
