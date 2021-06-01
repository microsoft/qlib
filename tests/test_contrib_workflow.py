# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import shutil
import unittest
from pathlib import Path

from qlib.contrib.workflow import MultiSegRecord, SignalMseRecord
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.tests import TestAutoData
from qlib.tests.config import CSI300_GBDT_TASK


def train_multiseg():
    model = init_instance_by_config(CSI300_GBDT_TASK["model"])
    dataset = init_instance_by_config(CSI300_GBDT_TASK["dataset"])
    with R.start(experiment_name="workflow"):
        R.log_params(**flatten_dict(CSI300_GBDT_TASK))
        model.fit(dataset)
        recorder = R.get_recorder()
        sr = MultiSegRecord(model, dataset, recorder)
        sr.generate(dict(valid="valid", test="test"), True)
        uri = R.get_uri()
    return uri


def train_mse():
    model = init_instance_by_config(CSI300_GBDT_TASK["model"])
    dataset = init_instance_by_config(CSI300_GBDT_TASK["dataset"])
    with R.start(experiment_name="workflow"):
        R.log_params(**flatten_dict(CSI300_GBDT_TASK))
        model.fit(dataset)
        recorder = R.get_recorder()
        sr = SignalMseRecord(recorder, model=model, dataset=dataset)
        sr.generate()
        uri = R.get_uri()
    return uri


class TestAllFlow(TestAutoData):
    def test_0_multiseg(self):
        uri_path = train_multiseg()
        shutil.rmtree(str(Path(uri_path.strip("file:")).resolve()))

    def test_1_mse(self):
        uri_path = train_mse()
        shutil.rmtree(str(Path(uri_path.strip("file:")).resolve()))


def suite():
    _suite = unittest.TestSuite()
    _suite.addTest(TestAllFlow("test_0_multiseg"))
    _suite.addTest(TestAllFlow("test_1_mse"))
    return _suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
