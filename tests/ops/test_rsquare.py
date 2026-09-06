"""Regression tests for Rsquare on degenerate (near-constant) windows.

See https://github.com/microsoft/qlib/issues/2297: the expanding (``N == 0``)
path leaked ``inf`` / garbage values on near-constant input because the NaN
guard existed only on the rolling path.
"""

import unittest
from unittest import mock

import numpy as np
import pandas as pd

from qlib.data.ops import Rsquare


def _fake_feature(values):
    """A feature whose load() returns a fixed series (no data infra needed)."""
    feature = mock.Mock()
    feature.load.return_value = pd.Series(values, dtype="float64")
    return feature


class TestRsquareDegenerateWindows(unittest.TestCase):
    def _assert_no_inf(self, series: pd.Series) -> None:
        self.assertEqual(np.isinf(series.values).sum(), 0, "Rsquare leaked inf values")

    def test_expanding_constant_run_is_masked(self):
        # A long constant run makes var(y) == 0 for every expanding window
        # fully inside it; those windows must be NaN, never inf/garbage.
        values = [1.0] * 60 + [2.0, 3.0, 4.0, 5.0]
        rs = Rsquare(_fake_feature(values), 0)
        out = rs._load_internal("unit", 0, len(values) - 1)
        self._assert_no_inf(out)
        # expanding windows of size >= 2 fully inside the constant run -> NaN
        self.assertTrue(bool(out.iloc[1:60].isna().all()), "degenerate expanding windows were not masked")
        # windows that include the varying tail see variance -> not masked
        self.assertFalse(bool(out.iloc[-3:].isna().all()))

    def test_expanding_normal_series_has_no_inf(self):
        # Sanity: ordinary varying input stays finite.
        values = np.sin(np.linspace(0, 6 * np.pi, 120)) + np.random.RandomState(0).normal(0, 1e-3, 120)
        rs = Rsquare(_fake_feature(values), 0)
        out = rs._load_internal("unit", 0, len(values) - 1)
        self._assert_no_inf(out)

    def test_rolling_constant_run_is_masked(self):
        # The rolling branch already had the guard; keep it green.
        values = [1.0] * 60 + [2.0, 3.0, 4.0, 5.0]
        rs = Rsquare(_fake_feature(values), 10)
        out = rs._load_internal("unit", 0, len(values) - 1)
        self._assert_no_inf(out)
        self.assertTrue(bool(out.iloc[10:59].isna().all()))


if __name__ == "__main__":
    unittest.main()
