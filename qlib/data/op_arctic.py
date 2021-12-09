# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import pandas as pd

from scipy.stats import percentileofscore

from .base import Expression, ExpressionOps, TExpression, TExpressionOps
from ..log import get_module_logger

try:
    from ._libs.rolling import rolling_slope, rolling_rsquare, rolling_resi
    from ._libs.expanding import expanding_slope, expanding_rsquare, expanding_resi
except ImportError as err:
    print("Do not import qlib package in the repository directory!")
    raise

__all__ = (
    "TRename",
    "TRolling",
    "TID",
    "TMax",
    "TResample",
    "TAdd",
    "TSub",
    "TMul",
    "TDiv",
    "TGreater",
    "TLess",
    "TAnd",
    "TOr",
    "TGt",
    "TGe",
    "TLt",
    "TLe",
    "TEq",
    "TNe",
    "TAbs",
    "TSign",
    "TLog",
    "TPower",
    "TMask",
    "TNot",
    "TRef",
    "TMean",
    "TStd",
    "TSum",
    "TVar",
    "TSkew",
    "TKurt",
    "TRsquare",
    "TResi",
    "TWMA",
    "TEMA",
    "TSlope",
    "TDelta",
    "TCount",
    "TRank",
    "TMad",
    "TMed",
    "TQuantile",
    "TIdxMin",
    "TIdxMax",
    "TMin",

)

#################### Resample ####################
class TID(TExpressionOps):
    def __init__(self, feature) -> None:
        self.feature = feature
        
    def __str__(self):
        return "{}({})".format(type(self).__name__, self.feature)
    
    def load_tick_data(self, instrument, start_index, end_index, freq, task_index):
        series = self.feature.load(instrument, start_index, end_index, freq)
        return series            
    
    def get_longest_back_rolling(self):
        return 0

    def get_extended_window_size(self):
        return 0, 0
    

class TElemOperator(TExpressionOps):
    """Element-wise Operator

    Parameters
    ----------
    feature : Expression
        feature instance
    func : str
        feature operation method

    Returns
    ----------
    Expression
        feature operation output
    """

    def __init__(self, feature, func):
        self.feature = feature
        self.func = func

    def __str__(self):
        return "{}({})".format(type(self).__name__, self.feature)

    def load_tick_data(self, instrument, start_index, end_index, freq, task_index):
        series = self.feature.load(instrument, start_index, end_index, freq)
        return getattr(np, self.func)(series)

    def get_longest_back_rolling(self):
        return self.feature.get_longest_back_rolling()

    def get_extended_window_size(self):
        return self.feature.get_extended_window_size()


class TAbs(TElemOperator):
    """Feature Absolute Value

    Parameters
    ----------
    feature : Expression
        feature instance

    Returns
    ----------
    Expression
        a feature instance with absolute output
    """

    def __init__(self, feature):
        super(TAbs, self).__init__(feature, "abs")


class TSign(TElemOperator):
    """Feature Sign

    Parameters
    ----------
    feature : Expression
        feature instance

    Returns
    ----------
    Expression
        a feature instance with sign
    """

    def __init__(self, feature):
        super(TSign, self).__init__(feature, "sign")


class TLog(TElemOperator):
    """Feature Log

    Parameters
    ----------
    feature : Expression
        feature instance

    Returns
    ----------
    Expression
        a feature instance with log
    """

    def __init__(self, feature):
        super(TLog, self).__init__(feature, "log")


class TPower(TElemOperator):
    """Feature Power

    Parameters
    ----------
    feature : Expression
        feature instance

    Returns
    ----------
    Expression
        a feature instance with power
    """

    def __init__(self, feature, exponent):
        super(TPower, self).__init__(feature, "power")
        self.exponent = exponent

    def __str__(self):
        return "{}({},{})".format(type(self).__name__, self.feature, self.exponent)

    def load_tick_data(self, instrument, start_index, end_index, freq, task_index):
        series = self.feature.load(instrument, start_index, end_index, freq)
        return getattr(np, self.func)(series, self.exponent)


class TMask(TElemOperator):
    """Feature Mask

    Parameters
    ----------
    feature : Expression
        feature instance
    instrument : str
        instrument mask

    Returns
    ----------
    Expression
        a feature instance with masked instrument
    """

    def __init__(self, feature, instrument):
        super(TMask, self).__init__(feature, "mask")
        self.instrument = instrument

    def __str__(self):
        return "{}({},{})".format(type(self).__name__, self.feature, self.instrument.lower())

    def load_tick_data(self, instrument, start_index, end_index, freq, task_index):
        return self.feature.load(self.instrument, start_index, end_index, freq)


class TNot(TElemOperator):
    """Not Operator

    Parameters
    ----------
    feature_left : Expression
        feature instance
    feature_right : Expression
        feature instance

    Returns
    ----------
    Feature:
        feature elementwise not output
    """

    def __init__(self, feature):
        super(TNot, self).__init__(feature, "bitwise_not")


#################### Resample ####################
class TResample(TExpressionOps):
    def __init__(self, feature, freq, func) -> None:
        self.feature = feature
        self.freq = freq
        self.func = func
        
    def __str__(self):
        return "{}({},{})".format(type(self).__name__, self.feature, self.freq)
    
    def load_tick_data(self, instrument, start_index, end_index, freq, task_index):
        #print("in load tick data:")
        series = self.feature.load(instrument, start_index, end_index, freq)
        #print("in load tick data:", len(series))
        if series.shape[0] == 0:
            return series
        else:
            if self.func == "sum":
                return getattr(series.resample(self.freq), self.func)(min_count=1).dropna(axis=0, how='any')  
            elif self.func == 'ffill':
                series = series[~series.index.duplicated(keep='first')]
                return getattr(series.resample(self.freq), self.func)().dropna(axis=0, how='any')
            else:
                return getattr(series.resample(self.freq), self.func)().dropna(axis=0, how='any')            
        
    def get_longest_back_rolling(self):
        return 0

    def get_extended_window_size(self):
        return 0, 0

class TRename(TExpressionOps):
    def __init__(self, feature, new_name):
        self.feature = feature
        #print("__init__, new_name: {}".format(new_name))
        self.new_name = new_name

    def __str__(self):
        return "{}({},{})".format(type(self).__name__, self.feature, self.new_name)

    def load_tick_data(self, instrument, start_index, end_index, freq, task_index):
        series = self.feature.load(instrument, start_index, end_index, freq, task_index)        
        series.name = self.new_name
        #print(series.name)
        return series
    
    def get_longest_back_rolling(self):
        return 0

    def get_extended_window_size(self):
        return 0, 0
    
#################### Rolling ####################
# NOTE: methods like `rolling.mean` are optimized with cython,
# and are super faster than `rolling.apply(np.mean)`
class TRolling(TExpressionOps):
    """Rolling Operator

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size
    func : str
        rolling method

    Returns
    ----------
    Expression
        rolling outputs
    """

    def __init__(self, feature, N, func):
        self.feature = feature
        self.N = N
        self.func = func

    def __str__(self):
        return "{}({},{})".format(type(self).__name__, self.feature, self.N)

    def load_tick_data(self, instrument, start_index, end_index, freq, task_index):
        series = self.feature.load(instrument, start_index, end_index, freq, task_index)
        # NOTE: remove all null check,
        # now it's user's responsibility to decide whether use features in null days
        # isnull = series.isnull() # NOTE: isnull = NaN, inf is not null
        #print("@@@@@debug, finish load, now will calculate")
        #print(series)
        ## resample at here
        if isinstance(self.N, int):
            if self.N == 0:
                series = getattr(series.expanding(min_periods=1), self.func)()
            elif 0 < self.N < 1:
                series = series.ewm(alpha=self.N, min_periods=1).mean()
            else:
                series = getattr(series.rolling(self.N, min_periods=1), self.func)()
            # series.iloc[:self.N-1] = np.nan
        # series[isnull] = np.nan
        else:
            series = getattr(series.rolling(self.N, min_periods=1), self.func)()
        #print("@@@debug finish caculate: {}".format(series.shape))
        return series

    def get_longest_back_rolling(self):
        if self.N == 0:
            return np.inf
        if 0 < self.N < 1:
            return int(np.log(1e-6) / np.log(1 - self.N))  # (1 - N)**window == 1e-6
        return self.feature.get_longest_back_rolling() + self.N - 1

    def get_extended_window_size(self):
        if self.N == 0:
            # FIXME: How to make this accurate and efficiently? Or  should we
            # remove such support for N == 0?
            get_module_logger(self.__class__.__name__).warning("The Rolling(ATTR, 0) will not be accurately calculated")
            return self.feature.get_extended_window_size()
        elif 0 < self.N < 1:
            lft_etd, rght_etd = self.feature.get_extended_window_size()
            size = int(np.log(1e-6) / np.log(1 - self.N))
            lft_etd = max(lft_etd + size - 1, lft_etd)
            return lft_etd, rght_etd
        else:
            lft_etd, rght_etd = self.feature.get_extended_window_size()
            lft_etd = max(lft_etd + self.N - 1, lft_etd)
            return lft_etd, rght_etd
    
class TMax(TRolling):
    def __init__(self, feature, N):
        super(TMax, self).__init__(feature, N, "max")

class TRef(TRolling):
    """Feature Reference

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        N = 0, retrieve the first data; N > 0, retrieve data of N periods ago; N < 0, future data

    Returns
    ----------
    Expression
        a feature instance with target reference
    """

    def __init__(self, feature, N):
        super(TRef, self).__init__(feature, N, "ref")

    def load_tick_data(self, instrument, start_index, end_index, freq, task_index):
        series = self.feature.load(instrument, start_index, end_index, freq)
        # N = 0, return first day
        if series.empty:
            return series  # Pandas bug, see: https://github.com/pandas-dev/pandas/issues/21049
        elif self.N == 0:
            series = pd.Series(series.iloc[0], index=series.index)
        else:
            series = series.shift(self.N)  # copy
        return series

    def get_longest_back_rolling(self):
        if self.N == 0:
            return np.inf
        return self.feature.get_longest_back_rolling() + self.N

    def get_extended_window_size(self):
        if self.N == 0:
            get_module_logger(self.__class__.__name__).warning("The Ref(ATTR, 0) will not be accurately calculated")
            return self.feature.get_extended_window_size()
        else:
            lft_etd, rght_etd = self.feature.get_extended_window_size()
            lft_etd = max(lft_etd + self.N, lft_etd)
            rght_etd = max(rght_etd - self.N, rght_etd)
            return lft_etd, rght_etd


class TMean(TRolling):
    """Rolling Mean (MA)

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with rolling average
    """

    def __init__(self, feature, N):
        super(TMean, self).__init__(feature, N, "mean")


class TSum(TRolling):
    """Rolling Sum

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with rolling sum
    """

    def __init__(self, feature, N):
        super(TSum, self).__init__(feature, N, "sum")


class TStd(TRolling):
    """Rolling Std

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with rolling std
    """

    def __init__(self, feature, N):
        super(TStd, self).__init__(feature, N, "std")


class TVar(TRolling):
    """Rolling Variance

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with rolling variance
    """

    def __init__(self, feature, N):
        super(TVar, self).__init__(feature, N, "var")


class TSkew(TRolling):
    """Rolling Skewness

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with rolling skewness
    """

    def __init__(self, feature, N):
        if N != 0 and N < 3:
            raise ValueError("The rolling window size of Skewness operation should >= 3")
        super(TSkew, self).__init__(feature, N, "skew")


class TKurt(TRolling):
    """Rolling Kurtosis

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with rolling kurtosis
    """

    def __init__(self, feature, N):
        if N != 0 and N < 4:
            raise ValueError("The rolling window size of Kurtosis operation should >= 5")
        super(TKurt, self).__init__(feature, N, "kurt")


class TIdxMax(TRolling):
    """Rolling Max Index

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with rolling max index
    """

    def __init__(self, feature, N):
        super(TIdxMax, self).__init__(feature, N, "idxmax")

    def load_tick_data(self, instrument, start_index, end_index, freq, task_index):
        series = self.feature.load(instrument, start_index, end_index, freq)
        if self.N == 0:
            series = series.expanding(min_periods=1).apply(lambda x: x.argmax() + 1, raw=True)
        else:
            series = series.rolling(self.N, min_periods=1).apply(lambda x: x.argmax() + 1, raw=True)
        return series


class TMin(TRolling):
    """Rolling Min

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with rolling min
    """

    def __init__(self, feature, N):
        super(TMin, self).__init__(feature, N, "min")


class TIdxMin(TRolling):
    """Rolling Min Index

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with rolling min index
    """

    def __init__(self, feature, N):
        super(TIdxMin, self).__init__(feature, N, "idxmin")

    def load_tick_data(self, instrument, start_index, end_index, freq, task_index):
        series = self.feature.load(instrument, start_index, end_index, freq)
        if self.N == 0:
            series = series.expanding(min_periods=1).apply(lambda x: x.argmin() + 1, raw=True)
        else:
            series = series.rolling(self.N, min_periods=1).apply(lambda x: x.argmin() + 1, raw=True)
        return series


class TQuantile(TRolling):
    """Rolling Quantile

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with rolling quantile
    """

    def __init__(self, feature, N, qscore):
        super(TQuantile, self).__init__(feature, N, "quantile")
        self.qscore = qscore

    def __str__(self):
        return "{}({},{},{})".format(type(self).__name__, self.feature, self.N, self.qscore)

    def load_tick_data(self, instrument, start_index, end_index, freq, task_index):
        series = self.feature.load(instrument, start_index, end_index, freq)
        if self.N == 0:
            series = series.expanding(min_periods=1).quantile(self.qscore)
        else:
            series = series.rolling(self.N, min_periods=1).quantile(self.qscore)
        return series


class TMed(TRolling):
    """Rolling Median

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with rolling median
    """

    def __init__(self, feature, N):
        super(TMed, self).__init__(feature, N, "median")


class TMad(TRolling):
    """Rolling Mean Absolute Deviation

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with rolling mean absolute deviation
    """

    def __init__(self, feature, N):
        super(TMad, self).__init__(feature, N, "mad")

    def load_tick_data(self, instrument, start_index, end_index, freq, task_index):
        series = self.feature.load(instrument, start_index, end_index, freq)
        # TODO: implement in Cython

        def mad(x):
            x1 = x[~np.isnan(x)]
            return np.mean(np.abs(x1 - x1.mean()))

        if self.N == 0:
            series = series.expanding(min_periods=1).apply(mad, raw=True)
        else:
            series = series.rolling(self.N, min_periods=1).apply(mad, raw=True)
        return series


class TRank(TRolling):
    """Rolling Rank (Percentile)

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with rolling rank
    """

    def __init__(self, feature, N):
        super(TRank, self).__init__(feature, N, "rank")

    def load_tick_data(self, instrument, start_index, end_index, freq, task_index):
        series = self.feature.load(instrument, start_index, end_index, freq)
        # TODO: implement in Cython

        def rank(x):
            if np.isnan(x[-1]):
                return np.nan
            x1 = x[~np.isnan(x)]
            if x1.shape[0] == 0:
                return np.nan
            return percentileofscore(x1, x1[-1]) / len(x1)

        if self.N == 0:
            series = series.expanding(min_periods=1).apply(rank, raw=True)
        else:
            series = series.rolling(self.N, min_periods=1).apply(rank, raw=True)
        return series


class TCount(TRolling):
    """Rolling Count

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with rolling count of number of non-NaN elements
    """

    def __init__(self, feature, N):
        super(TCount, self).__init__(feature, N, "count")


class TDelta(TRolling):
    """Rolling Delta

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with end minus start in rolling window
    """

    def __init__(self, feature, N):
        super(TDelta, self).__init__(feature, N, "delta")

    def load_tick_data(self, instrument, start_index, end_index, freq, task_index):
        series = self.feature.load(instrument, start_index, end_index, freq)
        if self.N == 0:
            series = series - series.iloc[0]
        else:
            series = series - series.shift(self.N)
        return series


# TODO:
# support pair-wise rolling like `Slope(A, B, N)`
class TSlope(TRolling):
    """Rolling Slope

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with regression slope of given window
    """

    def __init__(self, feature, N):
        super(TSlope, self).__init__(feature, N, "slope")

    def load_tick_data(self, instrument, start_index, end_index, freq, task_index):
        series = self.feature.load(instrument, start_index, end_index, freq)
        if self.N == 0:
            series = pd.Series(expanding_slope(series.values), index=series.index)
        else:
            series = pd.Series(rolling_slope(series.values, self.N), index=series.index)
        return series


class TRsquare(TRolling):
    """Rolling R-value Square

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with regression r-value square of given window
    """

    def __init__(self, feature, N):
        super(TRsquare, self).__init__(feature, N, "rsquare")

    def load_tick_data(self, instrument, start_index, end_index, freq, task_index):
        _series = self.feature.load(instrument, start_index, end_index, freq)
        if self.N == 0:
            series = pd.Series(expanding_rsquare(_series.values), index=_series.index)
        else:
            series = pd.Series(rolling_rsquare(_series.values, self.N), index=_series.index)
            series.loc[np.isclose(_series.rolling(self.N, min_periods=1).std(), 0, atol=2e-05)] = np.nan
        return series


class TResi(TRolling):
    """Rolling Regression Residuals

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with regression residuals of given window
    """

    def __init__(self, feature, N):
        super(TResi, self).__init__(feature, N, "resi")

    def load_tick_data(self, instrument, start_index, end_index, freq, task_index):
        series = self.feature.load(instrument, start_index, end_index, freq)
        if self.N == 0:
            series = pd.Series(expanding_resi(series.values), index=series.index)
        else:
            series = pd.Series(rolling_resi(series.values, self.N), index=series.index)
        return series


class TWMA(TRolling):
    """Rolling WMA

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with weighted moving average output
    """

    def __init__(self, feature, N):
        super(TWMA, self).__init__(feature, N, "wma")

    def load_tick_data(self, instrument, start_index, end_index, freq, task_index):
        series = self.feature.load(instrument, start_index, end_index, freq)
        # TODO: implement in Cython

        def weighted_mean(x):
            w = np.arange(len(x))
            w = w / w.sum()
            return np.nanmean(w * x)

        if self.N == 0:
            series = series.expanding(min_periods=1).apply(weighted_mean, raw=True)
        else:
            series = series.rolling(self.N, min_periods=1).apply(weighted_mean, raw=True)
        return series


class TEMA(TRolling):
    """Rolling Exponential Mean (EMA)

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int, float
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with regression r-value square of given window
    """

    def __init__(self, feature, N):
        super(TEMA, self).__init__(feature, N, "ema")

    def load_tick_data(self, instrument, start_index, end_index, freq, task_index):
        series = self.feature.load(instrument, start_index, end_index, freq)

        def exp_weighted_mean(x):
            a = 1 - 2 / (1 + len(x))
            w = a ** np.arange(len(x))[::-1]
            w /= w.sum()
            return np.nansum(w * x)

        if self.N == 0:
            series = series.expanding(min_periods=1).apply(exp_weighted_mean, raw=True)
        elif 0 < self.N < 1:
            series = series.ewm(alpha=self.N, min_periods=1).mean()
        else:
            series = series.ewm(span=self.N, min_periods=1).mean()
        return series



#################### Pair-Wise Operator ####################
class TPairOperator(TExpressionOps):
    """Pair-wise operator

    Parameters
    ----------
    feature_left : TExpression
        feature instance or numeric value
    feature_right : TExpression
        feature instance or numeric value
    func : str
        operator function

    Returns
    ----------
    Feature:
        two features' operation output
    """

    def __init__(self, feature_left, feature_right, func):
        self.feature_left = feature_left
        self.feature_right = feature_right
        self.func = func

    def __str__(self):
        return "{}({},{})".format(type(self).__name__, self.feature_left, self.feature_right)

    def load_tick_data(self, instrument, start_index, end_index, freq, task_index):
        assert any(
            [isinstance(self.feature_left, TExpression), self.feature_right, TExpression]
        ), "at least one of two inputs is Expression instance"
        
        if isinstance(self.feature_left, TExpression):
            series_left = self.feature_left.load(instrument, start_index, end_index, freq)
        else:
            series_left = self.feature_left  # numeric value
        if isinstance(self.feature_right, TExpression):
            series_right = self.feature_right.load(instrument, start_index, end_index, freq)
        else:
            series_right = self.feature_right
        return getattr(np, self.func)(series_left, series_right)

    def get_longest_back_rolling(self):
        if isinstance(self.feature_left, TExpression):
            left_br = self.feature_left.get_longest_back_rolling()
        else:
            left_br = 0

        if isinstance(self.feature_right, TExpression):
            right_br = self.feature_right.get_longest_back_rolling()
        else:
            right_br = 0
        return max(left_br, right_br)

    def get_extended_window_size(self):
        if isinstance(self.feature_left, TExpression):
            ll, lr = self.feature_left.get_extended_window_size()
        else:
            ll, lr = 0, 0

        if isinstance(self.feature_right, TExpression):
            rl, rr = self.feature_right.get_extended_window_size()
        else:
            rl, rr = 0, 0
        return max(ll, rl), max(lr, rr)

class TAdd(TPairOperator):
    """Add Operator

    Parameters
    ----------
    feature_left : TExpression
        feature instance
    feature_right : TExpression
        feature instance

    Returns
    ----------
    Feature:
        two features' sum
    """

    def __init__(self, feature_left, feature_right):
        super(TAdd, self).__init__(feature_left, feature_right, "add")


class TSub(TPairOperator):
    """Subtract Operator

    Parameters
    ----------
    feature_left : Expression
        feature instance
    feature_right : Expression
        feature instance

    Returns
    ----------
    Feature:
        two features' subtraction
    """

    def __init__(self, feature_left, feature_right):
        super(TSub, self).__init__(feature_left, feature_right, "subtract")


class TMul(TPairOperator):
    """Multiply Operator

    Parameters
    ----------
    feature_left : Expression
        feature instance
    feature_right : Expression
        feature instance

    Returns
    ----------
    Feature:
        two features' product
    """

    def __init__(self, feature_left, feature_right):
        super(TMul, self).__init__(feature_left, feature_right, "multiply")


class TDiv(TPairOperator):
    """Division Operator

    Parameters
    ----------
    feature_left : Expression
        feature instance
    feature_right : Expression
        feature instance

    Returns
    ----------
    Feature:
        two features' division
    """

    def __init__(self, feature_left, feature_right):
        super(TDiv, self).__init__(feature_left, feature_right, "divide")


class TGreater(TPairOperator):
    """Greater Operator

    Parameters
    ----------
    feature_left : Expression
        feature instance
    feature_right : Expression
        feature instance

    Returns
    ----------
    Feature:
        greater elements taken from the input two features
    """

    def __init__(self, feature_left, feature_right):
        super(TGreater, self).__init__(feature_left, feature_right, "maximum")


class TLess(TPairOperator):
    """Less Operator

    Parameters
    ----------
    feature_left : Expression
        feature instance
    feature_right : Expression
        feature instance

    Returns
    ----------
    Feature:
        smaller elements taken from the input two features
    """

    def __init__(self, feature_left, feature_right):
        super(TLess, self).__init__(feature_left, feature_right, "minimum")


class TGt(TPairOperator):
    """Greater Than Operator

    Parameters
    ----------
    feature_left : Expression
        feature instance
    feature_right : Expression
        feature instance

    Returns
    ----------
    Feature:
        bool series indicate `left > right`
    """

    def __init__(self, feature_left, feature_right):
        super(TGt, self).__init__(feature_left, feature_right, "greater")


class TGe(TPairOperator):
    """Greater Equal Than Operator

    Parameters
    ----------
    feature_left : Expression
        feature instance
    feature_right : Expression
        feature instance

    Returns
    ----------
    Feature:
        bool series indicate `left >= right`
    """

    def __init__(self, feature_left, feature_right):
        super(TGe, self).__init__(feature_left, feature_right, "greater_equal")


class TLt(TPairOperator):
    """Less Than Operator

    Parameters
    ----------
    feature_left : Expression
        feature instance
    feature_right : Expression
        feature instance

    Returns
    ----------
    Feature:
        bool series indicate `left < right`
    """

    def __init__(self, feature_left, feature_right):
        super(TLt, self).__init__(feature_left, feature_right, "less")


class TLe(TPairOperator):
    """Less Equal Than Operator

    Parameters
    ----------
    feature_left : Expression
        feature instance
    feature_right : Expression
        feature instance

    Returns
    ----------
    Feature:
        bool series indicate `left <= right`
    """

    def __init__(self, feature_left, feature_right):
        super(TLe, self).__init__(feature_left, feature_right, "less_equal")


class TEq(TPairOperator):
    """Equal Operator

    Parameters
    ----------
    feature_left : Expression
        feature instance
    feature_right : Expression
        feature instance

    Returns
    ----------
    Feature:
        bool series indicate `left == right`
    """

    def __init__(self, feature_left, feature_right):
        super(TEq, self).__init__(feature_left, feature_right, "equal")


class TNe(TPairOperator):
    """Not Equal Operator

    Parameters
    ----------
    feature_left : Expression
        feature instance
    feature_right : Expression
        feature instance

    Returns
    ----------
    Feature:
        bool series indicate `left != right`
    """

    def __init__(self, feature_left, feature_right):
        super(TNe, self).__init__(feature_left, feature_right, "not_equal")


class TAnd(TPairOperator):
    """And Operator

    Parameters
    ----------
    feature_left : Expression
        feature instance
    feature_right : Expression
        feature instance

    Returns
    ----------
    Feature:
        two features' row by row & output
    """

    def __init__(self, feature_left, feature_right):
        super(TAnd, self).__init__(feature_left, feature_right, "bitwise_and")


class TOr(TPairOperator):
    
    """Or Operator

    Parameters
    ----------
    feature_left : Expression
        feature instance
    feature_right : Expression
        feature instance

    Returns
    ----------
    Feature:
        two features' row by row | outputs
    """

    def __init__(self, feature_left, feature_right):
        super(TOr, self).__init__(feature_left, feature_right, "bitwise_or")