# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import unittest
import platform
import mlflow
import time
from pathlib import Path
import shutil


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
        # Measure repeated client construction, not one-time store creation,
        # plugin discovery, or the file-store deprecation warning machinery.
        mlflow.tracking.MlflowClient(tracking_uri=str(self.TMP_PATH))
        start = time.perf_counter()
        for i in range(10):
            _ = mlflow.tracking.MlflowClient(tracking_uri=str(self.TMP_PATH))
        end = time.perf_counter()
        elapsed = end - start
        if platform.system() == "Linux":
            self.assertLess(elapsed, 1e-2)  # it can be done in less than 10ms
        else:
            self.assertLess(elapsed, 2e-2)
        print(elapsed)


if __name__ == "__main__":
    unittest.main()
