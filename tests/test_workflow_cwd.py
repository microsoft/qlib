# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Regression tests for the current-working-directory assumptions in ``qlib.workflow``.

See https://github.com/microsoft/qlib/issues/2252:

1. ``MLflowExpManager`` built its ``FileLock`` path from ``file://`` URIs in a way that
   produced a CWD-relative path, so the lock (and its parent dirs) was created wherever
   the process happened to be running instead of at the location the URI named.
2. ``MLflowRecorder._log_uncommitted_code`` shelled out to ``git`` without capturing
   stderr; outside a git work tree git's usage banner leaked to the parent's stderr and
   each command emitted a noisy log record.
"""

import os
import shutil
import tempfile
import unittest
import subprocess
from contextlib import contextmanager
from pathlib import Path

import mlflow

from qlib.config import C
from qlib.workflow.expm import MLflowExpManager
from qlib.workflow.recorder import MLflowRecorder


@contextmanager
def chdir(path):
    """Temporarily change the working directory."""
    old = os.getcwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(old)


@contextmanager
def capture_fd_stderr():
    """
    Capture OS-level file descriptor 2.

    A child process spawned with ``shell=True`` and no ``stderr=`` redirection writes to
    the inherited fd 2, which ``contextlib.redirect_stderr`` (it only swaps ``sys.stderr``)
    would not see. We redirect the real fd so the test observes exactly what a user's
    terminal / log pipe would.
    """
    saved_fd = os.dup(2)
    tmp = tempfile.TemporaryFile(mode="w+b")
    os.dup2(tmp.fileno(), 2)
    try:
        yield tmp
    finally:
        os.dup2(saved_fd, 2)
        os.close(saved_fd)


class MLflowExpManagerCWDTest(unittest.TestCase):
    """The experiment FileLock must live at the absolute path the file:// URI names."""

    def setUp(self) -> None:
        self.store = Path(tempfile.mkdtemp(prefix="qlib-store-"))
        self.unrelated_cwd = Path(tempfile.mkdtemp(prefix="qlib-cwd-"))
        # Constructing MLflowExpManager mutates the global config; snapshot to restore it.
        self._saved_uri = C.exp_manager.get("kwargs", {}).get("uri")

    def tearDown(self) -> None:
        if self._saved_uri is not None:
            C.exp_manager.setdefault("kwargs", {})["uri"] = self._saved_uri
        shutil.rmtree(self.store, ignore_errors=True)
        shutil.rmtree(self.unrelated_cwd, ignore_errors=True)

    def test_filelock_resolves_to_absolute_uri_not_cwd(self):
        mlruns = self.store / "mlruns"  # absolute path that does not exist yet
        expm = MLflowExpManager(uri="file:" + str(mlruns), default_exp_name="Experiment")

        with chdir(self.unrelated_cwd):
            expm._get_or_create_exp(experiment_name="cwd-test")
            # The buggy code created `<cwd>/<abs-path-without-leading-slash>/filelock`,
            # i.e. a stray tree under the (unrelated) working directory. Nothing may land here.
            leftovers = sorted(p.name for p in Path.cwd().iterdir())
        self.assertEqual(
            leftovers,
            [],
            f"FileLock created files relative to CWD instead of the absolute URI: {leftovers}",
        )

        # And the lock/experiment store must exist at the absolute location the URI named.
        self.assertTrue(mlruns.exists(), f"experiment store was not created at {mlruns}")


class MLflowRecorderGitSnapshotTest(unittest.TestCase):
    """`_log_uncommitted_code` must be quiet outside a repo and still work inside one."""

    def setUp(self) -> None:
        self.store = Path(tempfile.mkdtemp(prefix="qlib-store-"))
        self.uri = "file:" + str(self.store / "mlruns")
        self.client = mlflow.tracking.MlflowClient(tracking_uri=self.uri)
        self.exp_id = self.client.create_experiment("git-snapshot-test")
        self.non_git = Path(tempfile.mkdtemp(prefix="qlib-nongit-"))
        self.git_repo = Path(tempfile.mkdtemp(prefix="qlib-gitrepo-"))

    def tearDown(self) -> None:
        for d in (self.store, self.non_git, self.git_repo):
            shutil.rmtree(d, ignore_errors=True)

    def _new_recorder(self) -> MLflowRecorder:
        rec = MLflowRecorder(self.exp_id, self.uri)
        run = rec.client.create_run(self.exp_id)
        rec.id = run.info.run_id
        return rec

    def test_non_git_cwd_is_silent_and_logs_no_artifacts(self):
        rec = self._new_recorder()
        with chdir(self.non_git), capture_fd_stderr() as err:
            rec._log_uncommitted_code()
            err.seek(0)
            leaked = err.read()

        # git's "Not a git repository" / "usage: git ..." banner must not leak to stderr.
        self.assertNotIn(b"Not a git repository", leaked)
        self.assertNotIn(b"usage: git", leaked)
        # No code snapshot artifacts should be produced outside a work tree.
        self.assertEqual([a.path for a in rec.client.list_artifacts(rec.id)], [])

    def test_git_work_tree_logs_code_snapshot(self):
        env = {
            **os.environ,
            "GIT_AUTHOR_NAME": "qlib-test",
            "GIT_AUTHOR_EMAIL": "qlib-test@example.com",
            "GIT_COMMITTER_NAME": "qlib-test",
            "GIT_COMMITTER_EMAIL": "qlib-test@example.com",
        }
        subprocess.run(["git", "init", "-q"], cwd=self.git_repo, check=True, env=env)
        (self.git_repo / "a.txt").write_text("hello\n")
        subprocess.run(["git", "add", "a.txt"], cwd=self.git_repo, check=True, env=env)
        (self.git_repo / "a.txt").write_text("hello\nworld\n")  # unstaged change on top

        rec = self._new_recorder()
        with chdir(self.git_repo):
            rec._log_uncommitted_code()

        artifacts = sorted(a.path for a in rec.client.list_artifacts(rec.id))
        self.assertEqual(artifacts, ["code_cached.txt", "code_diff.txt", "code_status.txt"])


if __name__ == "__main__":
    unittest.main()
