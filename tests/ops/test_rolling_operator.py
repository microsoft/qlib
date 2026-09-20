# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import unittest

import numpy as np
import pandas as pd

from qlib.data.base import Feature
from qlib.data.ops import IdxMax, IdxMin, Max, Min


class _SeriesFeature(Feature):
    """A leaf feature backed by an in-memory series.

    Only the data access is replaced; the rolling operators under test run their real code.
    """

    def __init__(self, name, values):
        super().__init__(name)
        self._values = values

    def _load_internal(self, instrument, start_index, end_index, freq):
        return pd.Series(self._values, dtype="float64")


class TestIdxRollingOperator(unittest.TestCase):
    """IdxMax/IdxMin must point at the value that Max/Min report for the same window.

    `$high` and `$low` are NaN whenever an instrument is suspended, so windows containing
    NaN are the normal case for the Alpha158 IMAX/IMIN/IMXD fields built on these operators.
    """

    WINDOW = 3
    VALUES = [1.0, 5.0, np.nan, 2.0, 3.0]

    def _load(self, op, name):
        feature = _SeriesFeature(name, self.VALUES)
        return op(feature, self.WINDOW).load("mock", 0, len(self.VALUES) - 1, "day")

    def _window(self, i):
        return np.asarray(self.VALUES[max(0, i - self.WINDOW + 1) : i + 1])

    def test_idxmax_points_at_the_maximum(self):
        idx = self._load(IdxMax, "idxmax_high")
        expected = self._load(Max, "idxmax_high_ref")
        for i in range(len(self.VALUES)):
            position = idx.iloc[i]
            self.assertFalse(np.isnan(position), f"IdxMax returned NaN at index {i}")
            value = self._window(i)[int(position) - 1]
            self.assertEqual(value, expected.iloc[i], f"IdxMax points at {value} but Max reports {expected.iloc[i]}")

    def test_idxmin_points_at_the_minimum(self):
        idx = self._load(IdxMin, "idxmin_low")
        expected = self._load(Min, "idxmin_low_ref")
        for i in range(len(self.VALUES)):
            position = idx.iloc[i]
            self.assertFalse(np.isnan(position), f"IdxMin returned NaN at index {i}")
            value = self._window(i)[int(position) - 1]
            self.assertEqual(value, expected.iloc[i], f"IdxMin points at {value} but Min reports {expected.iloc[i]}")


if __name__ == "__main__":
    unittest.main()
