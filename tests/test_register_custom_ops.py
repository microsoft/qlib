# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
import unittest
import numpy as np

import qlib
from qlib.data import D
from qlib.data.ops import Operators, ElemOperator, PairOperator
from qlib.config import REG_CN
from qlib.tests import TestAutoData


class Diff(ElemOperator):
    """Feature First Difference

    Parameters
    ----------
    feature : Expression
        feature instance

    Returns
    ----------
    Expression
        a feature instance with first difference
    """

    def __init__(self, feature):
        super(Diff, self).__init__(feature, "diff")

    def _load_internal(self, instrument, start_index, end_index, freq):
        series = self.feature.load(instrument, start_index, end_index, freq)
        return series.diff()

    def get_extended_window_size(self):
        lft_etd, rght_etd = self.feature.get_extended_window_size()
        return lft_etd + 1, rght_etd


class Distance(PairOperator):
    """Feature Distance

    Parameters
    ----------
    feature : Expression
        feature instance

    Returns
    ----------
    Expression
        a feature instance with distance
    """

    def __init__(self, feature_left, feature_right):
        super(Distance, self).__init__(feature_left, feature_right, "distance")

    def _load_internal(self, instrument, start_index, end_index, freq):
        series_left = self.feature_left.load(instrument, start_index, end_index, freq)
        series_right = self.feature_right.load(instrument, start_index, end_index, freq)
        return np.abs(series_left - series_right)

OpsList = [Diff, Distance]
Operators.register(OpsList)

class TestRegiterCustomOps(TestAutoData):
    def test_regiter_custom_ops(self):
        instruments = ["SH600000"]
        fields = ["Diff($close)", "Distance($close, Ref($close, 1))"]
        print(D.features(instruments, fields, start_time="2010-01-01", end_time="2017-12-31", freq="day"))


if __name__ == "__main__":
    unittest.main()
