# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import numpy as np
import pandas as pd
from datetime import datetime

from qlib.data.cache import H
from qlib.data.data import Cal
from qlib.data.ops import ElemOperator, PairOperator
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


def get_calendar_minute(freq="day", future=False):
    """Load High-Freq Calendar Minute Using Memcache"""
    flag = f"{freq}_future_{future}_day"
    if flag in H["c"]:
        _calendar = H["c"][flag]
    else:
        _calendar = np.array(list(map(lambda x: x.minute // 30, Cal.load_calendar(freq, future))))
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


class DayLast(ElemOperator):
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
        return series.groupby(_calendar[series.index]).transform("last")


class FFillNan(ElemOperator):
    """FFillNan Operator

    Parameters
    ----------
    feature : Expression
        feature instance

    Returns
    ----------
    feature:
        a forward fill nan feature
    """

    def _load_internal(self, instrument, start_index, end_index, freq):
        series = self.feature.load(instrument, start_index, end_index, freq)
        return series.fillna(method="ffill")


class BFillNan(ElemOperator):
    """BFillNan Operator

    Parameters
    ----------
    feature : Expression
        feature instance

    Returns
    ----------
    feature:
        a backfoward fill nan feature
    """

    def _load_internal(self, instrument, start_index, end_index, freq):
        series = self.feature.load(instrument, start_index, end_index, freq)
        return series.fillna(method="bfill")


class Date(ElemOperator):
    """Date Operator

    Parameters
    ----------
    feature : Expression
        feature instance

    Returns
    ----------
    feature:
        a series of that each value is the date corresponding to feature.index
    """

    def _load_internal(self, instrument, start_index, end_index, freq):
        _calendar = get_calendar_day(freq=freq)
        series = self.feature.load(instrument, start_index, end_index, freq)
        return pd.Series(_calendar[series.index], index=series.index)


class Select(PairOperator):
    """Select Operator

    Parameters
    ----------
    feature_left : Expression
        feature instance, select condition
    feature_right : Expression
        feature instance, select value

    Returns
    ----------
    feature:
        value(feature_right) that meets the condition(feature_left)

    """

    def _load_internal(self, instrument, start_index, end_index, freq):
        series_condition = self.feature_left.load(instrument, start_index, end_index, freq)
        series_feature = self.feature_right.load(instrument, start_index, end_index, freq)
        return series_feature.loc[series_condition]


class IsNull(ElemOperator):
    """IsNull Operator

    Parameters
    ----------
    feature : Expression
        feature instance

    Returns
    ----------
    feature:
        A series indicating whether the feature is nan
    """

    def _load_internal(self, instrument, start_index, end_index, freq):
        series = self.feature.load(instrument, start_index, end_index, freq)
        return series.isnull()


class IsInf(ElemOperator):
    """IsInf Operator

    Parameters
    ----------
    feature : Expression
        feature instance

    Returns
    ----------
    feature:
        A series indicating whether the feature is inf
    """

    def _load_internal(self, instrument, start_index, end_index, freq):
        series = self.feature.load(instrument, start_index, end_index, freq)
        return np.isinf(series)


class Cut(ElemOperator):
    """Cut Operator

    Parameters
    ----------
    feature : Expression
        feature instance
    l : int
        l > 0, delete the first l elements of feature (default is None, which means 0)
    r : int
        r < 0, delete the last -r elements of feature (default is None, which means 0)
    Returns
    ----------
    feature:
        A series with the first l and last -r elements deleted from the feature.
        Note: It is deleted from the raw data, not the sliced data
    """

    def __init__(self, feature, left=None, right=None):
        self.left = left
        self.right = right
        if (self.left is not None and self.left <= 0) or (self.right is not None and self.right >= 0):
            raise ValueError("Cut operator l shoud > 0 and r should < 0")

        super(Cut, self).__init__(feature)

    def _load_internal(self, instrument, start_index, end_index, freq):
        series = self.feature.load(instrument, start_index, end_index, freq)
        return series.iloc[self.left : self.right]

    def get_extended_window_size(self):
        ll = 0 if self.left is None else self.left
        rr = 0 if self.right is None else abs(self.right)
        lft_etd, rght_etd = self.feature.get_extended_window_size()
        lft_etd = lft_etd + ll
        rght_etd = rght_etd + rr
        return lft_etd, rght_etd
