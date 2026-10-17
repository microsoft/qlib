# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import copy

import pandas as pd
import pytest

from qlib.workflow.task.gen import RollingGen
from qlib.workflow.task.utils import TimeAdjuster


def _rolling(calendar, step):
    """Build a sliding RollingGen on a calendar, without qlib.init or market data."""
    adjuster = TimeAdjuster.__new__(TimeAdjuster)
    adjuster._future = True
    adjuster.cals = list(calendar)
    rolling = RollingGen.__new__(RollingGen)
    rolling.step = step
    rolling.rtype = RollingGen.ROLL_SD
    rolling.ds_extra_mod_func = None
    rolling.ta = adjuster
    rolling.test_key = "test"
    rolling.train_key = "train"
    rolling.trunc_days = None
    rolling.task_copy_func = copy.deepcopy
    return rolling


def _task(calendar, test_slice):
    return {
        "dataset": {
            "kwargs": {
                "segments": {
                    "train": (calendar[0], calendar[1]),
                    "valid": (calendar[2], calendar[3]),
                    "test": (calendar[test_slice.start], calendar[test_slice.stop - 1]),
                }
            }
        }
    }


def _test_segment(task):
    return task["dataset"]["kwargs"]["segments"]["test"]


def test_shift_returns_none_when_the_start_is_the_first_index_past_the_calendar():
    calendar = pd.bdate_range("2020-01-01", periods=6)
    rolling = _rolling(calendar, step=2)
    # indexes 4 and 5 are the last two days; step 2 lands the start on len(calendar).
    # ``get`` returns None there. ``> len(calendar)`` does not raise, so the
    # caller has to treat a missing test start as the end of rolling.
    assert rolling.ta.shift((calendar[4], calendar[5]), step=2, rtype=TimeAdjuster.SHIFT_SD) == (None, None)


def test_shift_keeps_an_end_past_the_calendar_when_the_start_is_valid():
    calendar = pd.bdate_range("2020-01-01", periods=6)
    rolling = _rolling(calendar, step=1)
    start, end = rolling.ta.shift((calendar[3], calendar[5]), step=1, rtype=TimeAdjuster.SHIFT_SD)
    assert start == calendar[4]
    assert end is None


def test_following_tasks_stop_when_the_next_window_starts_past_the_calendar():
    calendar = pd.bdate_range("2020-01-01", periods=6)
    rolling = _rolling(calendar, step=2)
    following = list(rolling.gen_following_tasks(_task(calendar, slice(4, 6)), calendar[-1]))
    assert following == []


def test_following_tasks_keep_windows_that_end_on_the_last_trading_day():
    calendar = pd.bdate_range("2020-01-01", periods=10)
    rolling = _rolling(calendar, step=2)
    following = list(rolling.gen_following_tasks(_task(calendar, slice(0, 2)), calendar[-1]))
    assert [_test_segment(task) for task in following] == [
        (calendar[2], calendar[3]),
        (calendar[4], calendar[5]),
        (calendar[6], calendar[7]),
        (calendar[8], calendar[9]),
    ]


def test_following_tasks_still_stop_when_the_next_start_passes_test_end():
    calendar = pd.bdate_range("2020-01-01", periods=10)
    rolling = _rolling(calendar, step=2)
    following = list(rolling.gen_following_tasks(_task(calendar, slice(0, 2)), calendar[5]))
    assert [_test_segment(task) for task in following] == [
        (calendar[2], calendar[3]),
        (calendar[4], calendar[5]),
    ]
