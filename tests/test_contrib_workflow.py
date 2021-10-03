# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from qlib.workflow.record_temp import SignalRecord
import shutil
import unittest
from pathlib import Path

from qlib.contrib.workflow import MultiSegRecord, SignalMseRecord
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.tests import TestAutoData
from qlib.tests.config import CSI300_GBDT_TASK


def train_multiseg(uri_path: str = None):
    model = init_instance_by_config(CSI300_GBDT_TASK["model"])
    dataset = init_instance_by_config(CSI300_GBDT_TASK["dataset"])
    with R.start(experiment_name="workflow", uri=uri_path):
        R.log_params(**flatten_dict(CSI300_GBDT_TASK))
        model.fit(dataset)
        recorder = R.get_recorder()
        sr = MultiSegRecord(model, dataset, recorder)
        sr.generate(dict(valid="valid", test="test"), True)
        uri = R.get_uri()
    return uri


def train_mse(uri_path: str = None):
    model = init_instance_by_config(CSI300_GBDT_TASK["model"])
    dataset = init_instance_by_config(CSI300_GBDT_TASK["dataset"])
    with R.start(experiment_name="workflow", uri=uri_path):
        R.log_params(**flatten_dict(CSI300_GBDT_TASK))
        model.fit(dataset)
        recorder = R.get_recorder()
        SignalRecord(recorder=recorder, model=model, dataset=dataset).generate()
        sr = SignalMseRecord(recorder)
        sr.generate()
        uri = R.get_uri()
    return uri


class TestAllFlow(TestAutoData):
    URI_PATH = "file:" + str(Path(__file__).parent.joinpath("test_contrib_mlruns").resolve())

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.URI_PATH.lstrip("file:"))

    def test_0_multiseg(self):
        uri_path = train_multiseg(self.URI_PATH)

    def test_1_mse(self):
        uri_path = train_mse(self.URI_PATH)


def suite():
    _suite = unittest.TestSuite()
    _suite.addTest(TestAllFlow("test_0_multiseg"))
    _suite.addTest(TestAllFlow("test_1_mse"))
    return _suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
