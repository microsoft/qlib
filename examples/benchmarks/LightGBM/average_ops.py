import math
import numpy as np

from qlib.data.ops import ElemOperator


class Avg(ElemOperator):
    MINUTES = 240

    def __init__(self, feature, start_index, end_index, func="nanmean"):
        assert start_index < end_index, "Avg in end_index must be greater than start_index"
        self.feature = feature
        self.s_i = start_index
        self.e_i = end_index
        self.func = func
        self.min_periods = 1 if self.func == "nanmean" else self.e_i - self.s_i
        super().__init__(feature)

    def _load_internal(self, instrument, start_index, end_index, freq):
        series = self.feature.load(instrument, start_index, end_index, freq)
        if series.empty:
            return series
        start_index = math.ceil(series.index[0] / self.MINUTES) * self.MINUTES
        res = series.rolling(self.e_i - self.s_i, min_periods=self.min_periods).mean()
        mask = []
        while start_index <= series.index[-1]:
            mask.append(start_index + self.e_i - 1)
            start_index += self.MINUTES
        res.loc[~series.index.isin(mask)] = np.nan
        return res

    def get_extended_window_size(self):
        lft_etd, rght_etd = self.feature.get_extended_window_size()
        return lft_etd + self.MINUTES, rght_etd + self.MINUTES

    def __str__(self):
        return "{}({},{},{},{})".format(type(self).__name__, self.feature, self.s_i, self.e_i, self.func)
