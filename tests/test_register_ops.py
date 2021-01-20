# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
import unittest
import numpy as np

import qlib
from qlib.data import D
from qlib.data.ops import ElemOperator, PairOperator
from qlib.config import REG_CN
from qlib.utils import exists_qlib_data
from qlib.tests import TestAutoData
from qlib.tests.data import GetData


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


class TestRegiterCustomOps(TestAutoData):
    @classmethod
    def setUpClass(cls) -> None:
        # use default data
        provider_uri = "~/.qlib/qlib_data/cn_data_simple"  # target_dir
        if not exists_qlib_data(provider_uri):
            print(f"Qlib data is not found in {provider_uri}")

            GetData().qlib_data(
                name="qlib_data_simple", region="cn", version="latest", interval="1d", target_dir=provider_uri
            )
        qlib.init(provider_uri=provider_uri, custom_ops=[Diff, Distance], region=REG_CN)

    def test_regiter_custom_ops(self):
        instruments = ["SH600000"]
        fields = ["Diff($close)", "Distance($close, Ref($close, 1))"]
        print(D.features(instruments, fields, start_time="2010-01-01", end_time="2017-12-31", freq="day"))


if __name__ == "__main__":
    unittest.main()
