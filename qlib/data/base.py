# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import division
from __future__ import print_function

import os
import abc
import pandas as pd
import numpy as np

from ..utils import code_to_fname
from ..log import get_module_logger


class Expression(abc.ABC):
    """
    Expression base class

    Expression is designed to handle the calculation of data with the format below
    data with two dimension for each instrument,
    - feature
    - time:  it  could be observation time or period time.
        - period time is designed for Point-in-time database.  For example, the period time maybe 2014Q4, its value can observed for multiple times(different value may be observed at different time due to amendment).
    """

    def __str__(self):
        return type(self).__name__

    def __repr__(self):
        return str(self)

    def __gt__(self, other):
        from .ops import Gt  # pylint: disable=C0415

        return Gt(self, other)

    def __ge__(self, other):
        from .ops import Ge  # pylint: disable=C0415

        return Ge(self, other)

    def __lt__(self, other):
        from .ops import Lt  # pylint: disable=C0415

        return Lt(self, other)

    def __le__(self, other):
        from .ops import Le  # pylint: disable=C0415

        return Le(self, other)

    def __eq__(self, other):
        from .ops import Eq  # pylint: disable=C0415

        return Eq(self, other)

    def __ne__(self, other):
        from .ops import Ne  # pylint: disable=C0415

        return Ne(self, other)

    def __add__(self, other):
        from .ops import Add  # pylint: disable=C0415

        return Add(self, other)

    def __radd__(self, other):
        from .ops import Add  # pylint: disable=C0415

        return Add(other, self)

    def __sub__(self, other):
        from .ops import Sub  # pylint: disable=C0415

        return Sub(self, other)

    def __rsub__(self, other):
        from .ops import Sub  # pylint: disable=C0415

        return Sub(other, self)

    def __mul__(self, other):
        from .ops import Mul  # pylint: disable=C0415

        return Mul(self, other)

    def __rmul__(self, other):
        from .ops import Mul  # pylint: disable=C0415

        return Mul(self, other)

    def __div__(self, other):
        from .ops import Div  # pylint: disable=C0415

        return Div(self, other)

    def __rdiv__(self, other):
        from .ops import Div  # pylint: disable=C0415

        return Div(other, self)

    def __truediv__(self, other):
        from .ops import Div  # pylint: disable=C0415

        return Div(self, other)

    def __rtruediv__(self, other):
        from .ops import Div  # pylint: disable=C0415

        return Div(other, self)

    def __pow__(self, other):
        from .ops import Power  # pylint: disable=C0415

        return Power(self, other)

    def __and__(self, other):
        from .ops import And  # pylint: disable=C0415

        return And(self, other)

    def __rand__(self, other):
        from .ops import And  # pylint: disable=C0415

        return And(other, self)

    def __or__(self, other):
        from .ops import Or  # pylint: disable=C0415

        return Or(self, other)

    def __ror__(self, other):
        from .ops import Or  # pylint: disable=C0415

        return Or(other, self)

    def load(self, instrument, start_index, end_index, *args):
        """load  feature
        This function is responsible for loading feature/expression based on the expression engine.

        The concerate implementation will be seperated by two parts
        1) caching data, handle errors.
            - This part is shared by all the expressions and implemented in Expression
        2) processing and calculating data based on the specific expression.
            - This part is different in each expression and implemented in each expression

        Expresion Engine is shared by different data.
        Different data will have different extra infomation for `args`.

        Parameters
        ----------
        instrument : str
            instrument code.
        start_index : str
            feature start index [in calendar].
        end_index : str
            feature end  index  [in calendar].

        *args may contains following information;
        1) if it is used in basic experssion engine data, it contains following arguments
            freq : str
                feature frequency.

        2) if is used in PIT data, it contains following arguments
            cur_pit:
                it is designed for the point-in-time data.

        Returns
        ----------
        pd.Series
            feature series: The index of the series is the calendar index
        """
        from .cache import H  # pylint: disable=C0415

        # cache
        cache_key = str(self), instrument, start_index, end_index, *args
        if cache_key in H["f"]:
            return H["f"][cache_key]
        if start_index is not None and end_index is not None and start_index > end_index:
            raise ValueError("Invalid index range: {} {}".format(start_index, end_index))
        try:
            series = self._load_internal(instrument, start_index, end_index, *args)
        except Exception as e:
            get_module_logger("data").debug(
                f"Loading data error: instrument={instrument}, expression={str(self)}, "
                f"start_index={start_index}, end_index={end_index}, args={args}. "
                f"error info: {str(e)}"
            )
            raise
        series.name = str(self)
        H["f"][cache_key] = series
        return series

    @abc.abstractmethod
    def _load_internal(self, instrument, start_index, end_index, *args) -> pd.Series:
        raise NotImplementedError("This function must be implemented in your newly defined feature")

    @abc.abstractmethod
    def get_longest_back_rolling(self):
        """Get the longest length of historical data the feature has accessed

        This is designed for getting the needed range of the data to calculate
        the features in specific range at first.  However, situations like
        Ref(Ref($close, -1), 1) can not be handled rightly.

        So this will only used for detecting the length of historical data needed.
        """
        # TODO: forward operator like Ref($close, -1) is not supported yet.
        raise NotImplementedError("This function must be implemented in your newly defined feature")

    @abc.abstractmethod
    def get_extended_window_size(self):
        """get_extend_window_size

        For to calculate this Operator in range[start_index, end_index]
        We have to get the *leaf feature* in
        range[start_index - lft_etd, end_index + rght_etd].

        Returns
        ----------
        (int, int)
            lft_etd, rght_etd
        """
        raise NotImplementedError("This function must be implemented in your newly defined feature")


class Feature(Expression):
    """Static Expression

    This kind of feature will load data from provider
    """

    def __init__(self, name=None):
        if name:
            self._name = name
        else:
            self._name = type(self).__name__

    def __str__(self):
        return "$" + self._name

    def _load_internal(self, instrument, start_index, end_index, freq):
        # load
        from .data import FeatureD  # pylint: disable=C0415

        return FeatureD.feature(instrument, str(self), start_index, end_index, freq)

    def get_longest_back_rolling(self):
        return 0

    def get_extended_window_size(self):
        return 0, 0


class PFeature(Feature):
    def __str__(self):
        return "$$" + self._name

    def _load_internal(self, instrument, start_index, end_index, cur_time):
        from .data import PITD  # pylint: disable=C0415

        return PITD.period_feature(instrument, str(self), start_index, end_index, cur_time)


class ExpressionOps(Expression):
    """Operator Expression

    This kind of feature will use operator for feature
    construction on the fly.
    """

    pass


class PExpression(abc.ABC):
    """PExpression base class"""

    def __str__(self):
        return type(self).__name__

    def __repr__(self):
        return str(self)

    def __gt__(self, other):
        if isinstance(other, Expression):
            from .ops import Gt

            return Gt(self, other)
        else:
            from .ops_period import PGt

            return PGt(self, other)

    def __ge__(self, other):
        if isinstance(other, Expression):
            from .ops import Ge

            return Ge(self, other)
        else:
            from .ops_period import PGe

            return PGe(self, other)

    def __lt__(self, other):
        if isinstance(other, Expression):
            from .ops import Lt

            return Lt(self, other)
        else:
            from .ops_period import PLt

            return PLt(self, other)

    def __le__(self, other):
        if isinstance(other, Expression):
            from .ops import Le

            return Le(self, other)
        else:
            from .ops_period import PLe

            return PLe(self, other)

    def __eq__(self, other):
        if isinstance(other, Expression):
            from .ops import Eq

            return Eq(self, other)
        else:
            from .ops_period import PEq

            return PEq(self, other)

    def __ne__(self, other):
        if isinstance(other, Expression):
            from .ops import Ne

            return Ne(self, other)
        else:
            from .ops_period import PNe

            return PNe(self, other)

    def __add__(self, other):
        if isinstance(other, Expression):
            from .ops import Add

            return Add(self, other)
        else:
            from .ops_period import PAdd

            return PAdd(self, other)

    def __radd__(self, other):
        if isinstance(other, Expression):
            from .ops import Add

            return Add(other, self)
        else:
            from .ops_period import PAdd

            return PAdd(other, self)

    def __sub__(self, other):
        if isinstance(other, Expression):
            from .ops import Sub

            return Sub(self, other)
        else:
            from .ops_period import PSub

            return PSub(self, other)

    def __rsub__(self, other):
        if isinstance(other, Expression):
            from .ops import Sub

            return Sub(other, self)
        else:
            from .ops_period import PSub

            return PSub(other, self)

    def __mul__(self, other):
        if isinstance(other, Expression):
            from .ops import Mul

            return Mul(self, other)
        else:
            from .ops_period import PMul

            return PMul(self, other)

    def __rmul__(self, other):
        if isinstance(other, Expression):
            from .ops import Mul

            return Mul(other, self)
        else:
            from .ops_period import PMul

            return PMul(other, self)

    def __div__(self, other):
        if isinstance(other, Expression):
            from .ops import Div

            return Div(self, other)
        else:
            from .ops_period import PDiv

            return PDiv(self, other)

    def __rdiv__(self, other):
        if isinstance(other, Expression):
            from .ops import Div

            return Div(other, self)
        else:
            from .ops_period import PDiv

            return PDiv(other, self)

    def __truediv__(self, other):
        if isinstance(other, Expression):
            from .ops import Div

            return Div(self, other)
        else:
            from .ops_period import PDiv

            return PDiv(self, other)

    def __rtruediv__(self, other):
        if isinstance(other, Expression):
            from .ops import Div

            return Div(other, self)
        else:
            from .ops_period import PDiv

            return PDiv(other, self)

    def __pow__(self, other):
        if isinstance(other, Expression):
            from .ops import Power

            return Power(self, other)
        else:
            from .ops_period import PPower

            return PPower(self, other)

    def __and__(self, other):
        if isinstance(other, Expression):
            from .ops import And

            return And(self, other)
        else:
            from .ops_period import PAnd

            return PAnd(self, other)

    def __rand__(self, other):
        if isinstance(other, Expression):
            from .ops import And

            return And(other, self)
        else:
            from .ops_period import PAnd

            return PAnd(other, self)

    def __or__(self, other):
        if isinstance(other, Expression):
            from .ops import Or

            return Or(self, other)
        else:
            from .ops_period import POr

            return POr(self, other)

    def __ror__(self, other):
        if isinstance(other, Expression):
            from .ops import Or

            return Or(other, self)
        else:
            from .ops_period import POr

            return POr(other, self)

    @abc.abstractmethod
    def load_period_data(self, instrument, start_offset, end_offset, cur_index, **kwargs):
        raise NotImplementedError("This function must be implemented in your newly defined feature")

    @abc.abstractmethod
    def get_period_offset(self, cur_index):
        raise NotImplementedError("This function must be implemented in your newly defined feature")

    def check_feature_exist(self, instrument):
        child_exist_list = [
            v.check_feature_exist(instrument) for k, v in self.__dict__.items() if isinstance(v, PExpression)
        ]
        return all(child_exist_list)

    def load(self, instrument, start_index, end_index, freq):

        if not self.check_feature_exist(instrument):
            get_module_logger("base").warning(f"WARN: period data not found for {str(self)}")
            return pd.Series(dtype="float32", name=str(self))

        from .cache import H

        # cache
        args = str(self), instrument, start_index, end_index, freq
        if args in H["f"]:
            return H["f"][args]
        if start_index is None or end_index is None or start_index > end_index:
            raise ValueError("Invalid index range: {} {}".format(start_index, end_index))

        from .data import Cal

        _calendar = Cal.calendar(freq=freq)
        resample_data = np.empty(end_index - start_index + 1, dtype="float32")

        for cur_index in range(start_index, end_index + 1):
            cur_time = _calendar[cur_index]
            # To load expression accurately, more historical data are required
            start_offset = self.get_period_offset(cur_index)
            # The calculated value will always the last element, so the end_offset is zero.
            try:
                resample_data[cur_index - start_index] = self.load_period_data(
                    instrument, start_offset, 0, cur_time, info=(start_index, end_index, cur_index)
                ).iloc[-1]
            except FileNotFoundError:
                get_module_logger("base").warning(f"WARN: period data not found for {str(self)}")
                return pd.Series(dtype="float32", name=str(self))

        resample_series = pd.Series(
            resample_data, index=pd.RangeIndex(start_index, end_index + 1), dtype="float32", name=str(self)
        )
        H["f"][args] = resample_series
        return resample_series

    def get_longest_back_rolling(self):
        return 0

    def get_extended_window_size(self):
        return 0, 0


# class PFeature(PExpression):
#     def __init__(self, name=None):
#         if name:
#             self._name = name.lower()
#         else:
#             self._name = type(self).__name__.lower()
#
#     def __str__(self):
#         return "$$" + self._name
#
#     def load_period_data(self, instrument, start_offset, end_offset, cur_index, **kwargs):
#         # BUG: cur_idnex is a date!!!!!
#         ### Zhou Code
#         from .data import PITD
#
#         return PITD.period_feature(instrument, str(self), start_offset, end_offset, cur_index, **kwargs)
#         # return pd.Series([1, 2, 3])  # fot test
#
#     def get_period_offset(self, cur_index):
#         return 0


class PExpressionOps(PExpression):
    """Operator Expression

    This kind of feature will use operator for feature
    construction on the fly.
    """

    pass
