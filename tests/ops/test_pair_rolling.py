# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import unittest

import numpy as np
import pandas as pd

from qlib.data.base import Expression
from qlib.data.ops import Corr


class StubFeature(Expression):
    """Minimal expression returning a fixed series, so operators can be tested without a data provider."""

    def __init__(self, values, name):
        self._values = list(values)
        self._name = name

    def __str__(self):
        # unique per stub: Expression.load caches series keyed on str(self)
        return self._name

    def _load_internal(self, instrument, start_index, end_index, *args):
        return pd.Series(self._values, dtype=float)

    def get_longest_back_rolling(self):
        return 0

    def get_extended_window_size(self):
        return 0, 0


class TestCorrExpanding(unittest.TestCase):
    """Corr(x, y, 0) must work in expanding mode instead of crashing on rolling(0)."""

    def test_corr_expanding_no_crash(self):
        left = [1.0, 2.0, 3.0, 4.0, 5.0]
        right = [2.0, 1.0, 4.0, 3.0, 6.0]
        res = Corr(StubFeature(left, "exp_l"), StubFeature(right, "exp_r"), 0).load("inst", 0, 4, "day")
        expected = pd.Series(left).expanding(min_periods=1).corr(pd.Series(right))
        np.testing.assert_allclose(res.values, expected.values)

    def test_corr_expanding_masks_constant_series(self):
        # while the left series is constant its std is 0, so corr must be masked to NaN
        left = [1.0, 1.0, 1.0, 2.0]
        right = [2.0, 1.0, 4.0, 3.0]
        res = Corr(StubFeature(left, "mask_l"), StubFeature(right, "mask_r"), 0).load("inst", 0, 3, "day")
        self.assertTrue(np.isnan(res.values[:3]).all())
        self.assertTrue(np.isfinite(res.values[3]))

    def test_corr_rolling_unchanged(self):
        left = [1.0, 2.0, 3.0, 4.0, 5.0]
        right = [2.0, 1.0, 4.0, 3.0, 6.0]
        res = Corr(StubFeature(left, "roll_l"), StubFeature(right, "roll_r"), 3).load("inst", 0, 4, "day")
        expected = pd.Series(left).rolling(3, min_periods=1).corr(pd.Series(right))
        np.testing.assert_allclose(res.values, expected.values)


if __name__ == "__main__":
    unittest.main()
