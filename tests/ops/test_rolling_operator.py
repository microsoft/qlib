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

    Both windows matter: `N > 0` takes the rolling branch, `N == 0` the expanding one, and
    each has its own call site. A test that only covers one leaves the other unpinned.
    """

    WINDOWS = (3, 0)
    VALUES = [1.0, 5.0, np.nan, 2.0, 3.0]

    def _load(self, op, window, name):
        feature = _SeriesFeature(f"{name}_{window}", self.VALUES)
        return op(feature, window).load("mock", 0, len(self.VALUES) - 1, "day")

    def _window(self, i, window):
        start = 0 if window == 0 else max(0, i - window + 1)
        return np.asarray(self.VALUES[start : i + 1])

    def _assert_points_at(self, idx_op, ref_op, name):
        for window in self.WINDOWS:
            idx = self._load(idx_op, window, name)
            expected = self._load(ref_op, window, f"{name}_ref")
            for i in range(len(self.VALUES)):
                with self.subTest(window=window, index=i):
                    position = idx.iloc[i]
                    self.assertFalse(np.isnan(position), f"{idx_op.__name__} returned NaN")
                    value = self._window(i, window)[int(position) - 1]
                    self.assertEqual(
                        value,
                        expected.iloc[i],
                        f"{idx_op.__name__} points at {value} but {ref_op.__name__} reports {expected.iloc[i]}",
                    )

    def test_idxmax_points_at_the_maximum(self):
        self._assert_points_at(IdxMax, Max, "idxmax_high")

    def test_idxmin_points_at_the_minimum(self):
        self._assert_points_at(IdxMin, Min, "idxmin_low")


if __name__ == "__main__":
    unittest.main()
