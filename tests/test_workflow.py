# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import unittest
from pathlib import Path
import shutil

from qlib.workflow import R
from qlib.tests import TestAutoData


class WorkflowTest(TestAutoData):
    # Creating the directory manually doesn't work with mlflow,
    # so we add a subfolder named .trash when we create the directory.
    TMP_PATH = Path("./.mlruns_tmp/.trash")

    def tearDown(self) -> None:
        if self.TMP_PATH.exists():
            shutil.rmtree(self.TMP_PATH)

    def test_get_local_dir(self):
        """ """
        self.TMP_PATH.mkdir(parents=True, exist_ok=True)

        with R.start(uri=str(self.TMP_PATH)):
            pass

        with R.uri_context(uri=str(self.TMP_PATH)):
            resume_recorder = R.get_recorder()
            resume_recorder.get_local_dir()


if __name__ == "__main__":
    unittest.main()
