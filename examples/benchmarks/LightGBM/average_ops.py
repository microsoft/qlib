#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import math
import numpy as np

from qlib.data.ops import ElemOperator


class Avg(ElemOperator):
    """On the 1min data, calculate the mean value of the specified range within the day

    Parameters
    ----------
    feature : Expression
        feature instance
    start_index: int
        start index, [0, 239)
    end_index: int
        end index, [1, 240]
    func: str
        value from ["nanmean", "mean"], same as "np.nanmean" or "np.mean", by default "nanmean"
    Notes
    ------
        start_index < end_index
    Examples
    ------
        close = [0, 1, 2, 3, 4, 5]
        Avg($close, 0, 2) == [np.nan, 0.5, np.nan, np.nan, np.nan, np.nan]
        Avg($close, 2, 4) == [np.nan, np.nan, np.nan, 2.5, np.nan, np.nan]

    Returns
    ----------
    Expression
        The data for each trading day is: data[end_index-1] = data[start_index: end_index]).mean()
    """

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
