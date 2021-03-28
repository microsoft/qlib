# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
import shutil
import unittest
from pathlib import Path

import qlib
from qlib.config import C
from qlib.contrib.workflow import MultiSegRecord, SignalMseRecord
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.tests import TestAutoData


market = "csi300"
benchmark = "SH000300"

###################################
# train model
###################################
data_handler_config = {
    "start_time": "2008-01-01",
    "end_time": "2020-08-01",
    "fit_start_time": "2008-01-01",
    "fit_end_time": "2014-12-31",
    "instruments": market,
}

task = {
    "model": {
        "class": "LGBModel",
        "module_path": "qlib.contrib.model.gbdt",
        "kwargs": {
            "loss": "mse",
            "colsample_bytree": 0.8879,
            "learning_rate": 0.0421,
            "subsample": 0.8789,
            "lambda_l1": 205.6999,
            "lambda_l2": 580.9768,
            "max_depth": 8,
            "num_leaves": 210,
            "num_threads": 20,
        },
    },
    "dataset": {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "Alpha158",
                "module_path": "qlib.contrib.data.handler",
                "kwargs": data_handler_config,
            },
            "segments": {
                "train": ("2008-01-01", "2014-12-31"),
                "valid": ("2015-01-01", "2016-12-31"),
                "test": ("2017-01-01", "2020-08-01"),
            },
        },
    },
}


def train_multiseg():
    model = init_instance_by_config(task["model"])
    dataset = init_instance_by_config(task["dataset"])
    with R.start(experiment_name="workflow"):
        R.log_params(**flatten_dict(task))
        model.fit(dataset)
        recorder = R.get_recorder()
        sr = MultiSegRecord(model, dataset, recorder)
        sr.generate(dict(valid="valid", test="test"), True)
        uri = R.get_uri()
    return uri


def train_mse():
    model = init_instance_by_config(task["model"])
    dataset = init_instance_by_config(task["dataset"])
    with R.start(experiment_name="workflow"):
        R.log_params(**flatten_dict(task))
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
