# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import copy
import unittest

import pandas as pd

from qlib.workflow.task.gen import RollingGen
from qlib.workflow.task.utils import TimeAdjuster


def _calendar(n=12):
    return list(pd.bdate_range("2020-01-01", periods=n))


def _adjuster(calendar):
    ta = TimeAdjuster.__new__(TimeAdjuster)
    ta._future = True
    ta.cals = calendar
    return ta


def _rolling(calendar):
    rg = RollingGen.__new__(RollingGen)
    rg.step = 2
    rg.rtype = RollingGen.ROLL_SD
    rg.ds_extra_mod_func = None
    rg.ta = _adjuster(calendar)
    rg.test_key = "test"
    rg.train_key = "train"
    rg.trunc_days = None
    rg.task_copy_func = copy.deepcopy
    rg.align_fit_window = True
    return rg


def _task(calendar):
    return {
        "dataset": {
            "kwargs": {
                "handler": {
                    "class": "Alpha158",
                    "kwargs": {
                        "fit_start_time": "2008-01-01",
                        "fit_end_time": "2020-08-01",
                    },
                },
                "segments": {
                    "train": (calendar[0], calendar[3]),
                    "valid": (calendar[4], calendar[5]),
                    "test": (calendar[6], calendar[8]),
                },
            }
        }
    }


class TestWalkForwardFitWindow(unittest.TestCase):
    def test_truncate_allows_the_first_trading_day(self):
        calendar = _calendar()
        ta = _adjuster(calendar)
        # test index 7 and trunc_days 3 clamps the end to index 4.
        # The train start is calendar index 0, which used to fail `tp_idx > 0`.
        truncated = ta.truncate((calendar[0], calendar[6]), calendar[7], days=3)

        self.assertEqual(truncated, (calendar[0], calendar[4]))

    def test_truncate_rejects_an_embargo_before_the_calendar(self):
        calendar = _calendar()
        ta = _adjuster(calendar)
        with self.assertRaises(ValueError):
            ta.truncate((calendar[0], calendar[2]), calendar[1], days=5)

    def test_rolling_fit_window_follows_the_train_segment(self):
        calendar = _calendar()
        rg = _rolling(calendar)
        task = _task(calendar)
        following = list(rg.gen_following_tasks(task, calendar[9]))

        self.assertEqual(len(following), 1)
        handler = following[0]["dataset"]["kwargs"]["handler"]["kwargs"]
        segments = following[0]["dataset"]["kwargs"]["segments"]
        self.assertEqual(handler["fit_start_time"], segments["train"][0])
        self.assertEqual(handler["fit_end_time"], segments["train"][1])
        self.assertLess(pd.Timestamp(handler["fit_end_time"]), pd.Timestamp(segments["test"][0]))
        self.assertEqual(segments["train"], (calendar[2], calendar[5]))

    def test_align_fit_window_can_be_disabled(self):
        calendar = _calendar()
        rg = _rolling(calendar)
        rg.align_fit_window = False
        task = _task(calendar)
        rg._update_task_segs(task, task["dataset"]["kwargs"]["segments"])
        handler = task["dataset"]["kwargs"]["handler"]["kwargs"]

        self.assertEqual(handler["fit_start_time"], "2008-01-01")
        self.assertEqual(handler["fit_end_time"], "2020-08-01")

    def test_string_handler_is_left_alone(self):
        calendar = _calendar()
        rg = _rolling(calendar)
        task = _task(calendar)
        task["dataset"]["kwargs"]["handler"] = "file:///tmp/handler.pkl"
        rg._update_task_segs(task, task["dataset"]["kwargs"]["segments"])

        self.assertEqual(task["dataset"]["kwargs"]["handler"], "file:///tmp/handler.pkl")


if __name__ == "__main__":
    unittest.main()
