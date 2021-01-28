import numpy as np
import pandas as pd
import importlib
from qlib.data.ops import ElemOperator, PairOperator
from qlib.config import C
from qlib.data.cache import H
from qlib.data.data import Cal


def get_calendar_day(freq="day", future=False):
    flag = f"{freq}_future_{future}_day"
    if flag in H["c"]:
        _calendar = H["c"][flag]
    else:
        _calendar = np.array(list(map(lambda x: x.date(), Cal.load_calendar(freq, future))))
        H["c"][flag] = _calendar
    return _calendar


class DayLast(ElemOperator):
    def _load_internal(self, instrument, start_index, end_index, freq):
        _calendar = get_calendar_day(freq=freq)
        series = self.feature.load(instrument, start_index, end_index, freq)
        return series.groupby(_calendar[series.index]).transform("last")


class FFillNan(ElemOperator):
    def _load_internal(self, instrument, start_index, end_index, freq):
        series = self.feature.load(instrument, start_index, end_index, freq)
        return series.fillna(method="ffill")


class BFillNan(ElemOperator):
    def _load_internal(self, instrument, start_index, end_index, freq):
        series = self.feature.load(instrument, start_index, end_index, freq)
        return series.fillna(method="bfill")


class Date(ElemOperator):
    def _load_internal(self, instrument, start_index, end_index, freq):
        _calendar = get_calendar_day(freq=freq)
        series = self.feature.load(instrument, start_index, end_index, freq)
        return pd.Series(_calendar[series.index], index=series.index)


class Select(PairOperator):
    def _load_internal(self, instrument, start_index, end_index, freq):
        series_condition = self.feature_left.load(instrument, start_index, end_index, freq)
        series_feature = self.feature_right.load(instrument, start_index, end_index, freq)
        return series_feature.loc[series_condition]


class IsNull(ElemOperator):
    def _load_internal(self, instrument, start_index, end_index, freq):
        series = self.feature.load(instrument, start_index, end_index, freq)
        return series.isnull()
