import numpy as np
import pandas as pd
import importlib
from qlib.data.ops import ElemOperator, PairOperator
from qlib.config import C
from qlib.data.data import Cal


class DayFirst(ElemOperator):
    def __init__(self, feature):
        super(DayFirst, self).__init__(feature, "day_first")

    def _load_internal(self, instrument, start_index, end_index, freq):
        _calendar = Cal.get_calendar_day(freq=freq)[0]
        series = self.feature.load(instrument, start_index, end_index, freq)
        return series.groupby(_calendar[series.index]).transform("first")


class DayLast(ElemOperator):
    def __init__(self, feature):
        super(DayLast, self).__init__(feature, "day_last")

    def _load_internal(self, instrument, start_index, end_index, freq):
        _calendar = Cal.get_calendar_day(freq=freq)[0]
        series = self.feature.load(instrument, start_index, end_index, freq)
        return series.groupby(_calendar[series.index]).transform("last")


class FFillNan(ElemOperator):
    def __init__(self, feature):
        super(FFillNan, self).__init__(feature, "fill_nan")

    def _load_internal(self, instrument, start_index, end_index, freq):
        series = self.feature.load(instrument, start_index, end_index, freq)
        return series.fillna(method="ffill")


class Date(ElemOperator):
    def __init__(self, feature):
        super(Date, self).__init__(feature, "date")

    def _load_internal(self, instrument, start_index, end_index, freq):
        _calendar = Cal.get_calendar_day(freq=freq)[0]
        series = self.feature.load(instrument, start_index, end_index, freq)
        return pd.Series(_calendar[series.index], index=series.index)


class Select(PairOperator):
    def __init__(self, condition, feature):
        super(Select, self).__init__(condition, feature, "select")

    def _load_internal(self, instrument, start_index, end_index, freq):
        series_condition = self.feature_left.load(instrument, start_index, end_index, freq)
        series_feature = self.feature_right.load(instrument, start_index, end_index, freq)
        return series_feature.loc[series_condition]


class IsNull(ElemOperator):
    def __init__(self, feature):
        super(IsNull, self).__init__(feature, "isnull")

    def _load_internal(self, instrument, start_index, end_index, freq):
        series = self.feature.load(instrument, start_index, end_index, freq)
        return series.isnull()
