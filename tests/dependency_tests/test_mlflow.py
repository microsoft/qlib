# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import unittest
import platform
import mlflow
import time
from pathlib import Path
import shutil
from qlib.workflow.expm import MLflowExpManager


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
            # This is a regression guard, not a microbenchmark. Shared CI
            # runners regularly exceed 10ms despite normal client behavior.
            self.assertLess(elapsed, 1e-1)
        else:
            self.assertLess(elapsed, 2e-2)
        print(elapsed)

    def test_mlflow_allow_file_store_default(self):
        previous = os.environ.pop("MLFLOW_ALLOW_FILE_STORE", None)
        try:
            manager = MLflowExpManager(str(self.TMP_PATH), "default_test_exp")
            _ = manager.client
            self.assertEqual(os.environ.get("MLFLOW_ALLOW_FILE_STORE"), "true")

            os.environ["MLFLOW_ALLOW_FILE_STORE"] = "1"
            _ = manager.client
            self.assertEqual(os.environ.get("MLFLOW_ALLOW_FILE_STORE"), "1")
        finally:
            if previous is None:
                os.environ.pop("MLFLOW_ALLOW_FILE_STORE", None)
            else:
                os.environ["MLFLOW_ALLOW_FILE_STORE"] = previous


if __name__ == "__main__":
    unittest.main()
