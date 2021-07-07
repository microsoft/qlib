import numpy as np
import pandas as pd
import importlib
from qlib.data.ops import ElemOperator, PairOperator
from qlib.config import C
from qlib.data.cache import H
from qlib.data.data import Cal


def get_calendar_day(freq="day", future=False):
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
        _calendar = np.array(list(map(lambda x: pd.Timestamp(x.date()), Cal.load_calendar(freq, future))))
        H["c"][flag] = _calendar
    return _calendar


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

    def __init__(self, feature, l=None, r=None):
        self.l = l
        self.r = r
        if (self.l is not None and self.l <= 0) or (self.r is not None and self.r >= 0):
            raise ValueError("Cut operator l shoud > 0 and r should < 0")

        super(Cut, self).__init__(feature)

    def _load_internal(self, instrument, start_index, end_index, freq):
        series = self.feature.load(instrument, start_index, end_index, freq)
        return series.iloc[self.l : self.r]

    def get_extended_window_size(self):
        ll = 0 if self.l is None else self.l
        rr = 0 if self.r is None else abs(self.r)
        lft_etd, rght_etd = self.feature.get_extended_window_size()
        lft_etd = lft_etd + ll
        rght_etd = rght_etd + rr
        return lft_etd, rght_etd
