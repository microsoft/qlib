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
    """Expression base class"""

    def __str__(self):
        return type(self).__name__

    def __repr__(self):
        return str(self)

    def __gt__(self, other):
        from .ops import Gt

        return Gt(self, other)

    def __ge__(self, other):
        from .ops import Ge

        return Ge(self, other)

    def __lt__(self, other):
        from .ops import Lt

        return Lt(self, other)

    def __le__(self, other):
        from .ops import Le

        return Le(self, other)

    def __eq__(self, other):
        from .ops import Eq

        return Eq(self, other)

    def __ne__(self, other):
        from .ops import Ne

        return Ne(self, other)

    def __add__(self, other):
        from .ops import Add

        return Add(self, other)

    def __radd__(self, other):
        from .ops import Add

        return Add(other, self)

    def __sub__(self, other):
        from .ops import Sub

        return Sub(self, other)

    def __rsub__(self, other):
        from .ops import Sub

        return Sub(other, self)

    def __mul__(self, other):
        from .ops import Mul

        return Mul(self, other)

    def __rmul__(self, other):
        from .ops import Mul

        return Mul(self, other)

    def __div__(self, other):
        from .ops import Div

        return Div(self, other)

    def __rdiv__(self, other):
        from .ops import Div

        return Div(other, self)

    def __truediv__(self, other):
        from .ops import Div

        return Div(self, other)

    def __rtruediv__(self, other):
        from .ops import Div

        return Div(other, self)

    def __pow__(self, other):
        from .ops import Power

        return Power(self, other)

    def __and__(self, other):
        from .ops import And

        return And(self, other)

    def __rand__(self, other):
        from .ops import And

        return And(other, self)

    def __or__(self, other):
        from .ops import Or

        return Or(self, other)

    def __ror__(self, other):
        from .ops import Or

        return Or(other, self)

    def load(self, instrument, start_index, end_index, freq):
        """load  feature

        Parameters
        ----------
        instrument : str
            instrument code.
        start_index : str
            feature start index [in calendar].
        end_index : str
            feature end  index  [in calendar].
        freq : str
            feature frequency.

        Returns
        ----------
        pd.Series
            feature series: The index of the series is the calendar index
        """
        from .cache import H

        # cache
        args = str(self), instrument, start_index, end_index, freq
        if args in H["f"]:
            return H["f"][args]
        if start_index is None or end_index is None or start_index > end_index:
            raise ValueError("Invalid index range: {} {}".format(start_index, end_index))
        series = self._load_internal(instrument, start_index, end_index, freq)
        series.name = str(self)
        H["f"][args] = series
        return series

    @abc.abstractmethod
    def _load_internal(self, instrument, start_index, end_index, freq):
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
            self._name = name.lower()
        else:
            self._name = type(self).__name__.lower()

    def __str__(self):
        return "$" + self._name

    def _load_internal(self, instrument, start_index, end_index, freq):
        # load
        from .data import FeatureD

        return FeatureD.feature(instrument, str(self), start_index, end_index, freq)

    def get_longest_back_rolling(self):
        return 0

    def get_extended_window_size(self):
        return 0, 0


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
            cur_date = _calendar[cur_index]
            start_offset = self.get_period_offset(cur_index)
            resample_data[cur_index - start_index] = self.load_period_data(
                instrument, start_offset, 0, cur_date, info=(start_index, end_index, cur_index)
            ).iloc[-1]

        resample_series = pd.Series(
            resample_data, index=pd.RangeIndex(start_index, end_index + 1), dtype="float32", name=str(self)
        )
        H["f"][args] = resample_series
        return resample_series

    def get_longest_back_rolling(self):
        return 0

    def get_extended_window_size(self):
        return 0, 0


class PFeature(PExpression):
    def __init__(self, name=None):
        if name:
            self._name = name.lower()
        else:
            self._name = type(self).__name__.lower()

    def __str__(self):
        return "$$" + self._name

    def check_feature_exist(self, instrument):
        from .data import FeatureD

        instrument = code_to_fname(instrument).lower()
        index_path = FeatureD.uri_period_index.format(instrument, self._name)
        data_path = FeatureD.uri_period_data.format(instrument, self._name)

        return os.path.exists(index_path) and os.path.exists(data_path)

    def load_period_data(self, instrument, start_offset, end_offset, cur_index, **kwargs):
        ### Zhou Code
        from .data import FeatureD

        return FeatureD.period_feature(instrument, str(self), start_offset, end_offset, cur_index, **kwargs)
        # return pd.Series([1, 2, 3])  # fot test

    def get_period_offset(self, cur_index):
        return 0


class PExpressionOps(PExpression):
    """Operator Expression

    This kind of feature will use operator for feature
    construction on the fly.
    """

    pass


class OpsWrapper:
    """Ops Wrapper"""

    def __init__(self):
        self._ops = {}

    def reset(self):
        self._ops = {}

    def register(self, ops_list):
        for operator in ops_list:
            if not issubclass(operator, ExpressionOps) and not issubclass(operator, PExpressionOps):
                raise TypeError("operator must be subclass of ExpressionOps or PExpressionOps, not {}".format(operator))

            if operator.__name__ in self._ops:
                get_module_logger(self.__class__.__name__).warning(
                    "The custom operator [{}] will override the qlib default definition".format(operator.__name__)
                )
            self._ops[operator.__name__] = operator

    def __getattr__(self, key):
        if key not in self._ops:
            raise AttributeError("The operator [{0}] is not registered".format(key))
        return self._ops[key]


Operators = OpsWrapper()


def register_all_ops(C):
    """register all operator"""
    logger = get_module_logger("base")

    Operators.reset()

    from .ops import OpsList
    from .ops_period import PeriodOpsList

    Operators.register(OpsList)
    Operators.register(PeriodOpsList)

    if getattr(C, "custom_ops", None) is not None:
        Operators.register(C.custom_ops)
        logger.debug("register custom period operator {}".format(C.custom_ops))
