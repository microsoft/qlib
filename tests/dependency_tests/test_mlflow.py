# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import unittest
import platform
import contextlib
import io
import mlflow
import os
import time
from pathlib import Path
import shutil
import tempfile

from qlib.workflow.expm import _file_uri_to_path
from qlib.workflow.recorder import MLflowRecorder


class DummyClient:
    def __init__(self):
        self.logged_text = []

    def log_text(self, run_id, text, fname):
        self.logged_text.append((run_id, text, fname))


class MLflowTest(unittest.TestCase):
    TMP_PATH = Path("./.mlruns_tmp/")

    def tearDown(self) -> None:
        if self.TMP_PATH.exists():
            shutil.rmtree(self.TMP_PATH)

    def test_creating_client(self):
        """
        Please refer to qlib/workflow/expm.py:MLflowExpManager._client
        we don't cache _client (this is helpful to reduce maintainance work when MLflowExpManager's uri is chagned)

        This implementation is based on the assumption creating a client is fast
        """
        start = time.time()
        for i in range(10):
            _ = mlflow.tracking.MlflowClient(tracking_uri=str(self.TMP_PATH))
        end = time.time()
        elapsed = end - start
        if platform.system() == "Linux":
            self.assertLess(elapsed, 1e-2)  # it can be done in less than 10ms
        else:
            self.assertLess(elapsed, 2e-2)
        print(elapsed)

    def test_file_uri_to_path_keeps_absolute_paths(self):
        self.assertEqual(_file_uri_to_path("file:///tmp/qlib/mlruns"), Path("/tmp/qlib/mlruns"))
        self.assertEqual(_file_uri_to_path("file:/tmp/qlib/mlruns"), Path("/tmp/qlib/mlruns"))

    def test_log_uncommitted_code_skips_non_git_cwd_quietly(self):
        recorder = object.__new__(MLflowRecorder)
        recorder.id = "run-id"
        recorder.client = DummyClient()
        stderr = io.StringIO()

        old_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                os.chdir(tmpdir)
                with contextlib.redirect_stderr(stderr):
                    recorder._log_uncommitted_code()
            finally:
                os.chdir(old_cwd)

        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(recorder.client.logged_text, [])


if __name__ == "__main__":
    unittest.main()
