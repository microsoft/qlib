# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import division
from __future__ import print_function

import abc
import pandas as pd
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

    def __rpow__(self, other):
        from .ops import Power  # pylint: disable=C0415

        return Power(other, self)

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

        The concrete implementation will be separated into two parts:

        1) caching data, handle errors.

            - This part is shared by all the expressions and implemented in Expression
        2) processing and calculating data based on the specific expression.

            - This part is different in each expression and implemented in each expression

        Expression Engine is shared by different data.
        Different data will have different extra information for `args`.

        Parameters
        ----------
        instrument : str
            instrument code.
        start_index : str
            feature start index [in calendar].
        end_index : str
            feature end  index  [in calendar].

        *args may contain following information:
        1) if it is used in basic expression engine data, it contains following arguments
            freq: str
                feature frequency.

        2) if is used in PIT data, it contains following arguments
            cur_pit:
                it is designed for the point-in-time data.
            period: int
                This is used for query specific period.
                The period is represented with int in Qlib. (e.g. 202001 may represent the first quarter in 2020)

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

    def _load_internal(self, instrument, start_index, end_index, cur_time, period=None):
        from .data import PITD  # pylint: disable=C0415

        return PITD.period_feature(instrument, str(self), start_index, end_index, cur_time, period)


class ExpressionOps(Expression):
    """Operator Expression

    This kind of feature will use operator for feature
    construction on the fly.
    """
