# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import division
from __future__ import print_function

import sys
import abc
import numpy as np
import pandas as pd

from scipy.stats import percentileofscore

from .base import PExpression, PExpressionOps
from ..log import get_module_logger

try:
    from ._libs.rolling import rolling_slope, rolling_rsquare, rolling_resi
    from ._libs.expanding import expanding_slope, expanding_rsquare, expanding_resi
except ImportError:
    print(
        "#### Do not import qlib package in the repository directory in case of importing qlib from . without compiling #####"
    )
    raise


np.seterr(invalid="ignore")

#################### Element-Wise Operator ####################


class PElemOperator(PExpressionOps):

    def __init__(self, feature):
        self.feature = feature

    def __str__(self):
        return "{}({})".format(type(self).__name__, self.feature)

    def get_period_offset(self, cur_index):
        return self.feature.get_period_offset(cur_index)


class PNpElemOperator(PElemOperator):
    """Numpy Element-wise Operator

    Parameters
    ----------
    feature : Expression
        feature instance
    func : str
        numpy feature operation method

    Returns
    ----------
    Expression
        feature operation output
    """

    def __init__(self, feature, func):
        self.func = func
        super(PNpElemOperator, self).__init__(feature)

    def load_period_data(self, instrument, start_offset, end_offset, cur_index):
        series = self.feature.load_period_data(instrument, start_offset, end_offset, cur_index)
        return getattr(np, self.func)(series)

class PAbs(PNpElemOperator):
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
        super(PAbs, self).__init__(feature, "abs")


class PSign(PNpElemOperator):
    """Feature PSign

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
        super(PSign, self).__init__(feature, "sign")

    def load_period_data(self, instrument, start_offset, end_offset, cur_index):
        """
        To avoid error raised by bool type input, we transform the data into float32.
        """
        series = self.feature.load_period_data(instrument, start_offset, end_offset, cur_index)
        # TODO:  More precision types should be configurable
        series = series.astype(np.float32)
        return getattr(np, self.func)(series)


class PLog(PNpElemOperator):
    """Feature PLog

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
        super(PLog, self).__init__(feature, "log")


class PPower(PNpElemOperator):
    """Feature PPower

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
        super(PPower, self).__init__(feature, "power")
        self.exponent = exponent

    def __str__(self):
        return "{}({},{})".format(type(self).__name__, self.feature, self.exponent)

    def load_period_data(self, instrument, start_offset, end_offset, cur_index):
        series = self.feature.load_period_data(instrument, start_offset, end_offset, cur_index)
        return getattr(np, self.func)(series, self.exponent)


class PMask(PNpElemOperator):
    """Feature PMask

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
        super(PMask, self).__init__(feature, "mask")
        self.instrument = instrument

    def __str__(self):
        return "{}({},{})".format(type(self).__name__, self.feature, self.instrument.lower())

    def load_period_data(self, instrument, start_offset, end_offset, cur_index):
        
        return self.feature.load_period_data(self.instrument, start_offset, end_offset, cur_index)


class PNot(PNpElemOperator):
    """PNot Operator

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
        super(PNot, self).__init__(feature, "bitwise_not")


#################### Pair-Wise Operator ####################
class PPairOperator(PExpressionOps):
    """Pair-wise operator

    Parameters
    ----------
    feature_left : Expression
        feature instance or numeric value
    feature_right : Expression
        feature instance or numeric value
    func : str
        operator function

    Returns
    ----------
    Feature:
        two features' operation output
    """

    def __init__(self, feature_left, feature_right):
        self.feature_left = feature_left
        self.feature_right = feature_right

    def __str__(self):
        return "{}({},{})".format(type(self).__name__, self.feature_left, self.feature_right)

    def get_period_offset(self, cur_index):
        if isinstance(self.feature_left, PExpression):
            left_br = self.feature_left.get_period_offset(cur_index)
        else:
            left_br = 0

        if isinstance(self.feature_right, PExpression):
            right_br = self.feature_right.get_period_offset(cur_index)
        else:
            right_br = 0
        return max(left_br, right_br)

class PNpPairOperator(PPairOperator):
    """Numpy Pair-wise operator

    Parameters
    ----------
    feature_left : Expression
        feature instance or numeric value
    feature_right : Expression
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
        super(PNpPairOperator, self).__init__(feature_left, feature_right)

    def load_period_data(self, instrument, start_offset, end_offset, cur_index):
        assert any(
            [isinstance(self.feature_left, Expression), self.feature_right, Expression]
        ), "at least one of two inputs is Expression instance"
        if isinstance(self.feature_left, Expression):
            series_left = self.feature_left.load_period_data(instrument, start_offset, end_offset, cur_index)
        else:
            series_left = self.feature_left  # numeric value
        if isinstance(self.feature_right, Expression):
            series_right = self.feature_right.load_period_data(instrument, start_offset, end_offset, cur_index)
        else:
            series_right = self.feature_right
        return getattr(np, self.func)(series_left, series_right)


class PAdd(PNpPairOperator):
    """PAdd Operator

    Parameters
    ----------
    feature_left : Expression
        feature instance
    feature_right : Expression
        feature instance

    Returns
    ----------
    Feature:
        two features' sum
    """

    def __init__(self, feature_left, feature_right):
        super(PAdd, self).__init__(feature_left, feature_right, "add")


class PSub(PNpPairOperator):
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
        super(PSub, self).__init__(feature_left, feature_right, "subtract")


class PMul(PNpPairOperator):
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
        super(PMul, self).__init__(feature_left, feature_right, "multiply")


class PDiv(PNpPairOperator):
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
        super(PDiv, self).__init__(feature_left, feature_right, "divide")


class PGreater(PNpPairOperator):
    """PGreater Operator

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
        super(PGreater, self).__init__(feature_left, feature_right, "maximum")


class PLess(PNpPairOperator):
    """PLess Operator

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
        super(PLess, self).__init__(feature_left, feature_right, "minimum")


class PGt(PNpPairOperator):
    """PGreater Than Operator

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
        super(PGt, self).__init__(feature_left, feature_right, "greater")


class PGe(PNpPairOperator):
    """PGreater Equal Than Operator

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
        super(PGe, self).__init__(feature_left, feature_right, "greater_equal")


class PLt(PNpPairOperator):
    """PLess Than Operator

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
        super(PLt, self).__init__(feature_left, feature_right, "less")


class PLe(PNpPairOperator):
    """PLess Equal Than Operator

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
        super(PLe, self).__init__(feature_left, feature_right, "less_equal")


class PEq(PNpPairOperator):
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
        super(PEq, self).__init__(feature_left, feature_right, "equal")


class PNe(PNpPairOperator):
    """PNot Equal Operator

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
        super(PNe, self).__init__(feature_left, feature_right, "not_equal")


class PAnd(PNpPairOperator):
    """PAnd Operator

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
        super(PAnd, self).__init__(feature_left, feature_right, "bitwise_and")


class POr(PNpPairOperator):
    """POr Operator

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
        super(POr, self).__init__(feature_left, feature_right, "bitwise_or")


#################### Triple-wise Operator ####################
class PIf(PExpressionOps):
    """PIf Operator

    Parameters
    ----------
    condition : Expression
        feature instance with bool values as condition
    feature_left : Expression
        feature instance
    feature_right : Expression
        feature instance
    """

    def __init__(self, condition, feature_left, feature_right):
        self.condition = condition
        self.feature_left = feature_left
        self.feature_right = feature_right

    def __str__(self):
        return "PIf({},{},{})".format(self.condition, self.feature_left, self.feature_right)

    def load_period_data(self, instrument, start_offset, end_offset, cur_index):
        series_cond = self.condition.load_period_data(instrument, start_offset, end_offset, cur_index)
        if isinstance(self.feature_left, Expression):
            series_left = self.feature_left.load_period_data(instrument, start_offset, end_offset, cur_index)
        else:
            series_left = self.feature_left
        if isinstance(self.feature_right, Expression):
            series_right = self.feature_right.load_period_data(instrument, start_offset, end_offset, cur_index)
        else:
            series_right = self.feature_right
        series = pd.Series(np.where(series_cond, series_left, series_right), index=series_cond.index)
        return series

    def get_period_offset(self, cur_index):
        if isinstance(self.feature_left, Expression):
            left_br = self.feature_left.get_period_offset(cur_index)
        else:
            left_br = 0

        if isinstance(self.feature_right, Expression):
            right_br = self.feature_right.get_period_offset(cur_index)
        else:
            right_br = 0

        if isinstance(self.condition, Expression):
            c_br = self.condition.get_period_offset(cur_index)
        else:
            c_br = 0
        return max(left_br, right_br, c_br)



#################### PRolling ####################
# NOTE: methods like `rolling.mean` are optimized with cython,
# and are super faster than `rolling.apply(np.mean)`


class PRolling(PExpressionOps):
    """PRolling Operator

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

    def load_period_data(self, instrument, start_offset, end_offset, cur_index):
        series = self.feature.load_period_data(instrument, start_offset, end_offset, cur_index)
        # NOTE: remove all null check,
        # now it's user's responsibility to decide whether use features in null days
        # isnull = series.isnull() # NOTE: isnull = NaN, inf is not null
        if self.N == 0:
            series = getattr(series.expanding(min_periods=1), self.func)()
        elif 0 < self.N < 1:
            series = series.ewm(alpha=self.N, min_periods=1).mean()
        else:
            series = getattr(series.rolling(self.N, min_periods=1), self.func)()
            # series.iloc[:self.N-1] = np.nan
        # series[isnull] = np.nan
        return series

    def get_period_offset(self, cur_index):
        if self.N == 0:
            return np.inf
        if 0 < self.N < 1:
            return int(np.log(1e-6) / np.log(1 - self.N))  # (1 - N)**window == 1e-6
        return self.feature.get_period_offset(cur_index) + self.N - 1


class PRef(PRolling):
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
        super(PRef, self).__init__(feature, N, "ref")

    def load_period_data(self, instrument, start_offset, end_offset, cur_index):
        series = self.feature.load_period_data(instrument, start_offset, end_offset, cur_index)
        # N = 0, return first day
        if series.empty:
            return series  # Pandas bug, see: https://github.com/pandas-dev/pandas/issues/21049
        elif self.N == 0:
            series = pd.Series(series.iloc[0], index=series.index)
        else:
            series = series.shift(self.N)  # copy
        return series

    def get_period_offset(self, cur_index):
        if self.N == 0:
            return np.inf
        return self.feature.get_period_offset(cur_index) + self.N

class PMean(PRolling):
    """PRolling PMean (MA)

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
        super(PMean, self).__init__(feature, N, "mean")


class PSum(PRolling):
    """PRolling PSum

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
        super(PSum, self).__init__(feature, N, "sum")


class PStd(PRolling):
    """PRolling PStd

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
        super(PStd, self).__init__(feature, N, "std")


class PVar(PRolling):
    """PRolling Variance

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
        super(PVar, self).__init__(feature, N, "var")


class PSkew(PRolling):
    """PRolling Skewness

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
        super(PSkew, self).__init__(feature, N, "skew")


class PKurt(PRolling):
    """PRolling Kurtosis

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
        super(PKurt, self).__init__(feature, N, "kurt")


class PMax(PRolling):
    """PRolling PMax

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with rolling max
    """

    def __init__(self, feature, N):
        super(PMax, self).__init__(feature, N, "max")


class PIdxMax(PRolling):
    """PRolling PMax Index

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
        super(PIdxMax, self).__init__(feature, N, "idxmax")

    def load_period_data(self, instrument, start_offset, end_offset, cur_index):
        series = self.feature.load_period_data(instrument, start_offset, end_offset, cur_index)
        if self.N == 0:
            series = series.expanding(min_periods=1).apply(lambda x: x.argmax() + 1, raw=True)
        else:
            series = series.rolling(self.N, min_periods=1).apply(lambda x: x.argmax() + 1, raw=True)
        return series


class PMin(PRolling):
    """PRolling PMin

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
        super(PMin, self).__init__(feature, N, "min")


class PIdxMin(PRolling):
    """PRolling PMin Index

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
        super(PIdxMin, self).__init__(feature, N, "idxmin")

    def load_period_data(self, instrument, start_offset, end_offset, cur_index):
        series = self.feature.load_period_data(instrument, start_offset, end_offset, cur_index)
        if self.N == 0:
            series = series.expanding(min_periods=1).apply(lambda x: x.argmin() + 1, raw=True)
        else:
            series = series.rolling(self.N, min_periods=1).apply(lambda x: x.argmin() + 1, raw=True)
        return series


class PQuantile(PRolling):
    """PRolling PQuantile

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
        super(PQuantile, self).__init__(feature, N, "quantile")
        self.qscore = qscore

    def __str__(self):
        return "{}({},{},{})".format(type(self).__name__, self.feature, self.N, self.qscore)

    def load_period_data(self, instrument, start_offset, end_offset, cur_index):
        series = self.feature.load_period_data(instrument, start_offset, end_offset, cur_index)
        if self.N == 0:
            series = series.expanding(min_periods=1).quantile(self.qscore)
        else:
            series = series.rolling(self.N, min_periods=1).quantile(self.qscore)
        return series


class PMed(PRolling):
    """PRolling Median

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
        super(PMed, self).__init__(feature, N, "median")


class PMad(PRolling):
    """PRolling PMean Absolute Deviation

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
        super(PMad, self).__init__(feature, N, "mad")

    def load_period_data(self, instrument, start_offset, end_offset, cur_index):
        series = self.feature.load_period_data(instrument, start_offset, end_offset, cur_index)
        # TODO: implement in Cython

        def mad(x):
            x1 = x[~np.isnan(x)]
            return np.mean(np.abs(x1 - x1.mean()))

        if self.N == 0:
            series = series.expanding(min_periods=1).apply(mad, raw=True)
        else:
            series = series.rolling(self.N, min_periods=1).apply(mad, raw=True)
        return series


class PRank(PRolling):
    """PRolling PRank (Percentile)

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
        super(PRank, self).__init__(feature, N, "rank")

    def load_period_data(self, instrument, start_offset, end_offset, cur_index):
        series = self.feature.load_period_data(instrument, start_offset, end_offset, cur_index)
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


class PCount(PRolling):
    """PRolling PCount

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
        super(PCount, self).__init__(feature, N, "count")


class PDelta(PRolling):
    """PRolling PDelta

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
        super(PDelta, self).__init__(feature, N, "delta")

    def load_period_data(self, instrument, start_offset, end_offset, cur_index):
        series = self.feature.load_period_data(instrument, start_offset, end_offset, cur_index)
        if self.N == 0:
            series = series - series.iloc[0]
        else:
            series = series - series.shift(self.N)
        return series


# TODO:
# support pair-wise rolling like `PSlope(A, B, N)`
class PSlope(PRolling):
    """PRolling PSlope

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with linear regression slope of given window
    """

    def __init__(self, feature, N):
        super(PSlope, self).__init__(feature, N, "slope")

    def load_period_data(self, instrument, start_offset, end_offset, cur_index):
        series = self.feature.load_period_data(instrument, start_offset, end_offset, cur_index)
        if self.N == 0:
            series = pd.Series(expanding_slope(series.values), index=series.index)
        else:
            series = pd.Series(rolling_slope(series.values, self.N), index=series.index)
        return series


class PRsquare(PRolling):
    """PRolling R-value Square

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with linear regression r-value square of given window
    """

    def __init__(self, feature, N):
        super(PRsquare, self).__init__(feature, N, "rsquare")

    def load_period_data(self, instrument, start_offset, end_offset, cur_index):
        _series = self.feature.load_period_data(instrument, start_offset, end_offset, cur_index)
        if self.N == 0:
            series = pd.Series(expanding_rsquare(_series.values), index=_series.index)
        else:
            series = pd.Series(rolling_rsquare(_series.values, self.N), index=_series.index)
            series.loc[np.isclose(_series.rolling(self.N, min_periods=1).std(), 0, atol=2e-05)] = np.nan
        return series


class PResi(PRolling):
    """PRolling Regression Residuals

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
        super(PResi, self).__init__(feature, N, "resi")

    def load_period_data(self, instrument, start_offset, end_offset, cur_index):
        series = self.feature.load_period_data(instrument, start_offset, end_offset, cur_index)
        if self.N == 0:
            series = pd.Series(expanding_resi(series.values), index=series.index)
        else:
            series = pd.Series(rolling_resi(series.values, self.N), index=series.index)
        return series


class PWMA(PRolling):
    """PRolling PWMA

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
        super(PWMA, self).__init__(feature, N, "wma")

    def load_period_data(self, instrument, start_offset, end_offset, cur_index):
        series = self.feature.load_period_data(instrument, start_offset, end_offset, cur_index)
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


class PEMA(PRolling):
    """PRolling Exponential PMean (PEMA)

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
        super(PEMA, self).__init__(feature, N, "ema")

    def load_period_data(self, instrument, start_offset, end_offset, cur_index):
        series = self.feature.load_period_data(instrument, start_offset, end_offset, cur_index)

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


#################### Pair-Wise PRolling ####################
class PairRolling(PExpressionOps):
    """Pair PRolling Operator

    Parameters
    ----------
    feature_left : Expression
        feature instance
    feature_right : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with rolling output of two input features
    """

    def __init__(self, feature_left, feature_right, N, func):
        self.feature_left = feature_left
        self.feature_right = feature_right
        self.N = N
        self.func = func

    def __str__(self):
        return "{}({},{},{})".format(type(self).__name__, self.feature_left, self.feature_right, self.N)

    def load_period_data(self, instrument, start_offset, end_offset, cur_index):
        series_left = self.feature_left.load_period_data(instrument, start_offset, end_offset, cur_index)
        series_right = self.feature_right.load_period_data(instrument, start_offset, end_offset, cur_index)
        if self.N == 0:
            series = getattr(series_left.expanding(min_periods=1), self.func)(series_right)
        else:
            series = getattr(series_left.rolling(self.N, min_periods=1), self.func)(series_right)
        return series

    def get_period_offset(self, cur_index):
        if self.N == 0:
            return np.inf
        return (
            max(self.feature_left.get_period_offset(cur_index), self.feature_right.get_period_offset(cur_index))
            + self.N
            - 1
        )

    def get_extended_window_size(self):
        if self.N == 0:
            get_module_logger(self.__class__.__name__).warning(
                "The PairRolling(ATTR, 0) will not be accurately calculated"
            )
            return self.feature.get_extended_window_size()
        else:
            ll, lr = self.feature_left.get_extended_window_size()
            rl, rr = self.feature_right.get_extended_window_size()
            return max(ll, rl) + self.N - 1, max(lr, rr)


class PCorr(PairRolling):
    """PRolling Correlation

    Parameters
    ----------
    feature_left : Expression
        feature instance
    feature_right : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with rolling correlation of two input features
    """

    def __init__(self, feature_left, feature_right, N):
        super(PCorr, self).__init__(feature_left, feature_right, N, "corr")

    def load_period_data(self, instrument, start_offset, end_offset, cur_index):
        res = super(PCorr, self)._load_internal(instrument, start_index, end_index, freq)

        # NOTE: Load uses MemCache, so calling load_period_data again will not cause performance degradation
        series_left = self.feature_left.load_period_data(instrument, start_offset, end_offset, cur_index)
        series_right = self.feature_right.load_period_data(instrument, start_offset, end_offset, cur_index)
        res.loc[
            np.isclose(series_left.rolling(self.N, min_periods=1).std(), 0, atol=2e-05)
            | np.isclose(series_right.rolling(self.N, min_periods=1).std(), 0, atol=2e-05)
        ] = np.nan
        return res


class PCov(PairRolling):
    """PRolling Covariance

    Parameters
    ----------
    feature_left : Expression
        feature instance
    feature_right : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with rolling max of two input features
    """

    def __init__(self, feature_left, feature_right, N):
        super(PCov, self).__init__(feature_left, feature_right, N, "cov")


OpsList = [
    PRef,
    PMax,
    PMin,
    PSum,
    PMean,
    PStd,
    PVar,
    PSkew,
    PKurt,
    PMed,
    PMad,
    PSlope,
    PRsquare,
    PResi,
    PRank,
    PQuantile,
    PCount,
    PEMA,
    PWMA,
    PCorr,
    PCov,
    PDelta,
    PAbs,
    PSign,
    PLog,
    PPower,
    PAdd,
    PSub,
    PMul,
    PDiv,
    PGreater,
    PLess,
    PAnd,
    POr,
    PNot,
    PGt,
    PGe,
    PLt,
    PLe,
    PEq,
    PNe,
    PMask,
    PIdxMax,
    PIdxMin,
    PIf,
]


def register_all_period_ops(C):
    """register all operator"""
    logger = get_module_logger("ops")

    from .base import Operators

    Operators.reset()
    Operators.register(OpsList)

    if getattr(C, "custom_period_ops", None) is not None:
        Operators.register(C.custom_ops)
        logger.debug("register custom period operator {}".format(C.custom_ops))
