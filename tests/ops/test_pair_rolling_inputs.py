# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import unittest

import numpy as np
import pandas as pd

from qlib.data.base import Expression
from qlib.data.ops import Cov


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


class TestPairRollingInputCheck(unittest.TestCase):
    """PairRolling must reject two non-Expression inputs instead of failing later with AttributeError."""

    def test_two_numeric_inputs_rejected(self):
        with self.assertRaises(AssertionError):
            Cov(1.0, 2.0, 3).load("inst", 0, 4, "day")

    def test_expression_inputs_accepted(self):
        left = [1.0, 2.0, 3.0, 4.0, 5.0]
        right = [2.0, 1.0, 4.0, 3.0, 6.0]
        res = Cov(StubFeature(left, "cov_l"), StubFeature(right, "cov_r"), 3).load("inst", 0, 4, "day")
        expected = pd.Series(left).rolling(3, min_periods=1).cov(pd.Series(right))
        np.testing.assert_allclose(res.values, expected.values)


if __name__ == "__main__":
    unittest.main()
