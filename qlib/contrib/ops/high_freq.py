# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from pathlib import Path
import numpy as np
import pandas as pd

import qlib
from qlib.data import D
from qlib.data.cache import H
from qlib.data.data import Cal
from qlib.data.ops import ElemOperator


def get_calendar_day(freq="1min", future=False):
    """Load High-Freq Calendar Date Using Memcache.

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
    """DayLast Operator

    Parameters
    ----------
    feature : Expression
        feature instance

    Returns
    ----------
    feature:
        a series of that each value equals the last value of its day
    """

    def _load_internal(self, instrument, start_index, end_index, freq):
        _calendar = get_calendar_day(freq=freq)
        series = self.feature.load(instrument, start_index, end_index, freq)
        return series.groupby(_calendar[series.index]).cumsum()
