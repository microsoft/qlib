# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import division
from __future__ import print_function

import abc
import pandas as pd


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
        pass

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


class TExpression(abc.ABC):
    """TExpression base class"""

    def __str__(self):
        return type(self).__name__

    def __repr__(self):
        return str(self)

    def __gt__(self, other):
        if isinstance(other, Expression):
            from .ops import Gt

            return Gt(self, other)
        else:
            from .ops_arctic import TGt

            return TGt(self, other)

    def __ge__(self, other):
        if isinstance(other, Expression):
            from .ops import Ge

            return Ge(self, other)
        else:
            from .ops_arctic import TGe

            return TGe(self, other)

    def __lt__(self, other):
        if isinstance(other, Expression):
            from .ops import Lt

            return Lt(self, other)
        else:
            from .ops_arctic import TLt

            return TLt(self, other)

    def __le__(self, other):
        if isinstance(other, Expression):
            from .ops import Le

            return Le(self, other)
        else:
            from .ops_arctic import TLe

            return TLe(self, other)

    def __eq__(self, other):
        if isinstance(other, Expression):
            from .ops import Eq

            return Eq(self, other)
        else:
            from .ops_arctic import TEq

            return TEq(self, other)

    def __ne__(self, other):
        if isinstance(other, Expression):
            from .ops import Ne

            return Ne(self, other)
        else:
            from .ops_arctic import TNe

            return TNe(self, other)

    def __add__(self, other):
        if isinstance(other, Expression):
            from .ops import Add

            return Add(self, other)
        else:
            from .ops_arctic import TAdd

            return TAdd(self, other)

    def __radd__(self, other):
        if isinstance(other, Expression):
            from .ops import Add

            return Add(other, self)
        else:
            from .ops_arctic import TAdd

            return TAdd(other, self)

    def __sub__(self, other):
        if isinstance(other, Expression):
            from .ops import Sub

            return Sub(self, other)
        else:
            from .ops_arctic import TSub

            return TSub(self, other)

    def __rsub__(self, other):
        if isinstance(other, Expression):
            from .ops import Sub

            return Sub(other, self)
        else:
            from .ops_arctic import TSub

            return TSub(other, self)

    def __mul__(self, other):
        if isinstance(other, Expression):
            from .ops import Mul

            return Mul(self, other)
        else:
            from .ops_arctic import TMul

            return TMul(self, other)

    def __rmul__(self, other):
        if isinstance(other, Expression):
            from .ops import Mul

            return Mul(other, self)
        else:
            from .ops_arctic import TMul

            return TMul(other, self)

    def __div__(self, other):
        if isinstance(other, Expression):
            from .ops import Div

            return Div(self, other)
        else:
            from .ops_arctic import TDiv

            return TDiv(self, other)

    def __rdiv__(self, other):
        if isinstance(other, Expression):
            from .ops import Div

            return Div(other, self)
        else:
            from .ops_arctic import TDiv

            return TDiv(other, self)

    def __truediv__(self, other):
        if isinstance(other, Expression):
            from .ops import Div

            return Div(self, other)
        else:
            from .ops_arctic import TDiv

            return TDiv(self, other)

    def __rtruediv__(self, other):
        if isinstance(other, Expression):
            from .ops import Div

            return Div(other, self)
        else:
            from .ops_arctic import TDiv

            return TDiv(other, self)

    def __pow__(self, other):
        if isinstance(other, Expression):
            from .ops import Power

            return Power(self, other)
        else:
            from .ops_arctic import TPower

            return TPower(self, other)

    def __and__(self, other):
        if isinstance(other, Expression):
            from .ops import And

            return And(self, other)
        else:
            from .ops_arctic import TAnd

            return TAnd(self, other)

    def __rand__(self, other):
        if isinstance(other, Expression):
            from .ops import And

            return And(other, self)
        else:
            from .ops_arctic import TAnd

            return TAnd(other, self)

    def __or__(self, other):
        if isinstance(other, Expression):
            from .ops import Or

            return Or(self, other)
        else:
            from .ops_arctic import TOr

            return TOr(self, other)

    def __ror__(self, other):
        if isinstance(other, Expression):
            from .ops import Or

            return Or(other, self)
        else:
            from .ops_arctic import TOr

            return TOr(other, self)

    @abc.abstractmethod
    def load_tick_data(instrument, start_index, end_index, freq):
        raise NotImplementedError("This function must be implemented in your newly defined feature")

    def check_feature_exist(self, instrument):
        pass

    def load(self, instrument, start_index, end_index, freq):
        if start_index is None or end_index is None or start_index > end_index:
            raise ValueError("Invalid index range: {} {}".format(start_index, end_index))
        series = self.load_tick_data(instrument, start_index, end_index, freq)
        series.name = str(self)
        print("@@@@@debug finish load, shape is:{}, self is {}".format(series.shape, self))
        return series

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

class TFeature(TExpression):
    def __init__(self, name=None):
        if name:
            self._name = name.lower()
        else:
            self._name = type(self).__name__.lower()

    def __str__(self):
        return "@" + self._name

    def check_feature_exist(self, instrument):
        from .data import FeatureD
        pass

    def load_tick_data(self, instrument, start_index, end_index, freq):
        from .data import FeatureD
        print("@@@debug 4", instrument, start_index, end_index)
        return FeatureD.tick_feature(instrument, self._name, start_index, end_index, freq)


    def get_longest_back_rolling(self):
        return 0

    def get_extended_window_size(self):
        return 0, 0

class TExpressionOps(TExpression):
    """Operator Expression

    This kind of feature will use operator for feature
    construction on the fly.
    """

    pass