import copy
import unittest

import fire
import pandas as pd

import qlib
from qlib.config import REG_CN
from qlib.data import D
from qlib.model.trainer import task_train
from qlib.tests import TestAutoData
from qlib.tests.config import CSI300_GBDT_TASK
from qlib.workflow.online.utils import OnlineToolR


class TestRolling(TestAutoData):
    _setup_kwargs = dict(expression_cache=None, dataset_cache=None)

    def test_update_pred(self):
        task = copy.deepcopy(CSI300_GBDT_TASK)

        task["record"] = {
            "class": "SignalRecord",
            "module_path": "qlib.workflow.record_temp",
        }

        exp_name = "online_srv_test"

        cal = D.calendar()
        latest_date = cal[-1]

        train_start = latest_date - pd.Timedelta(days=61)
        train_end = latest_date - pd.Timedelta(days=41)
        task["dataset"]["kwargs"]["segments"] = {
            "train": (train_start, train_end),
            "valid": (latest_date - pd.Timedelta(days=40), latest_date - pd.Timedelta(days=21)),
            "test": (latest_date - pd.Timedelta(days=20), latest_date),
        }

        task["dataset"]["kwargs"]["handler"]["kwargs"] = {
            "start_time": train_start,
            "end_time": latest_date,
            "fit_start_time": train_start,
            "fit_end_time": train_end,
            "instruments": "csi300",
        }

        rec = task_train(task, exp_name)

        pred = rec.load_object("pred.pkl")

        online_tool = OnlineToolR(exp_name)
        online_tool.reset_online_tag(rec)  # set to online model

        online_tool.update_online_pred(to_date=latest_date + pd.Timedelta(days=10))


if __name__ == "__main__":
    unittest.main()
