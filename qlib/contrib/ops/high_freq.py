# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime

import qlib
from qlib.data import D
from qlib.data.cache import H
from qlib.data.data import Cal
from qlib.data.ops import ElemOperator
from qlib.utils.time import time_to_day_index


def get_calendar_day(freq="1min", future=False):
    """
    Load High-Freq Calendar Date Using Memcache.
    !!!NOTE: Loading the calendar is quite slow. So loading calendar before start multiprocessing will make it faster.

    Parameters
    ----------
    freq : str
        frequency of read calendar file.
    future : bool
        whether including future trading day.

    Returns
    -------
    _calendar:
        array of date.
    """
    flag = f"{freq}_future_{future}_day"
    if flag in H["c"]:
        _calendar = H["c"][flag]
    else:
        _calendar = np.array(list(map(lambda x: x.date(), Cal.load_calendar(freq, future))))
        H["c"][flag] = _calendar
    return _calendar


class DayCumsum(ElemOperator):
    """DayCumsum Operator during start time and end time.

    Parameters
    ----------
    feature : Expression
        feature instance
    start : str
        the start time of backtest in one day.
        !!!NOTE: "9:30" means the time period of (9:30, 9:31) is in transaction.
    end : str
        the end time of backtest in one day.
        !!!NOTE: "14:59" means the time period of (14:59, 15:00) is in transaction,
                but (15:00, 15:01) is not.
        So start="9:30" and end="14:59" means trading all day.

    Returns
    ----------
    feature:
        a series of that each value equals the cumsum value during start time and end time.
        Otherwise, the value is zero.
    """

    def __init__(self, feature, start: str = "9:30", end: str = "14:59"):
        self.feature = feature
        self.start = datetime.strptime(start, "%H:%M")
        self.end = datetime.strptime(end, "%H:%M")

        self.morning_open = datetime.strptime("9:30", "%H:%M")
        self.morning_close = datetime.strptime("11:30", "%H:%M")
        self.noon_open = datetime.strptime("13:00", "%H:%M")
        self.noon_close = datetime.strptime("15:00", "%H:%M")

        self.start_id = time_to_day_index(self.start)
        self.end_id = time_to_day_index(self.end)

    def period_cusum(self, df):
        df = df.copy()
        assert len(df) == 240
        df.iloc[0 : self.start_id] = 0
        df = df.cumsum()
        df.iloc[self.end_id + 1 : 240] = 0
        return df

    def _load_internal(self, instrument, start_index, end_index, freq):
        _calendar = get_calendar_day(freq=freq)
        series = self.feature.load(instrument, start_index, end_index, freq)
        return series.groupby(_calendar[series.index]).transform(self.period_cusum)
