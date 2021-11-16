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
from qlib.workflow.online.update import LabelUpdater


class TestRolling(TestAutoData):
    def test_update_pred(self):
        """
        This test is for testing if it will raise error if the `to_date` is out of the boundary.
        """
        task = copy.deepcopy(CSI300_GBDT_TASK)

        task["record"] = {
            "class": "SignalRecord",
            "module_path": "qlib.workflow.record_temp",
            "kwargs": {"dataset": "<DATASET>", "model": "<MODEL>"},
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

    def test_update_label(self):

        task = copy.deepcopy(CSI300_GBDT_TASK)

        task["record"] = {
            "class": "SignalRecord",
            "module_path": "qlib.workflow.record_temp",
            "kwargs": {"dataset": "<DATASET>", "model": "<MODEL>"},
        }

        exp_name = "online_srv_test"

        cal = D.calendar()
        shift = 10
        latest_date = cal[-1 - shift]

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
        online_tool.update_online_pred()

        new_pred = rec.load_object("pred.pkl")
        label = rec.load_object("label.pkl")
        label_date = label.dropna().index.get_level_values("datetime").max()
        pred_date = new_pred.dropna().index.get_level_values("datetime").max()

        # The prediction is updated, but the label is not updated.
        self.assertTrue(label_date < pred_date)

        # Update label now
        lu = LabelUpdater(rec)
        lu.update()
        new_label = rec.load_object("label.pkl")
        new_label_date = new_label.index.get_level_values("datetime").max()
        self.assertTrue(new_label_date == pred_date)  # make sure the label is updated now


if __name__ == "__main__":
    unittest.main()
