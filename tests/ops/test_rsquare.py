# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import unittest

import numpy as np
import pandas as pd

from qlib.data.ops import Rsquare


class _StubFeature:
    """Minimal Expression stub: returns a fixed series from ``load`` so that
    Rolling operators can be exercised without a data provider / market data."""

    def __init__(self, series: pd.Series):
        self._series = series

    def load(self, *args, **kwargs) -> pd.Series:
        return self._series


class TestRsquareDegenerateWindow(unittest.TestCase):
    """Regression tests for Rsquare on (near-)constant windows.

    A regression of a (near-)constant series is degenerate: the coefficient of
    determination is 0/0. The Cython kernels compute it as ``num / sqrt(var_x *
    var_y)`` where ``var_y ~ 0``, so floating-point cancellation yields ``inf``
    or a spurious finite value instead of ``NaN``. ``Rsquare`` guards against
    this by masking windows whose std is ~0 to ``NaN`` -- this must hold for the
    expanding (``N == 0``) path as well as the rolling (``N != 0``) path.
    """

    # near-constant: only float-noise variation, std ~ 5e-7 (< the 2e-5 guard atol)
    SERIES = pd.Series([100.0, 100.0, 100.0, 100.000001, 100.0, 100.0])

    def _rsquare(self, n: int) -> pd.Series:
        return Rsquare(_StubFeature(self.SERIES), n)._load_internal("SH600000", None, None)

    def test_expanding_window_is_guarded(self):
        # N == 0 -> expanding. Before the fix this leaked inf / garbage R^2.
        out = self._rsquare(0)
        self.assertFalse(np.isinf(out.to_numpy()).any(), f"expanding Rsquare leaked inf: {out.tolist()}")
        finite = out.dropna()
        self.assertTrue(((finite >= 0.0) & (finite <= 1.0)).all(), f"R^2 outside [0, 1]: {finite.tolist()}")

    def test_rolling_window_is_guarded(self):
        out = self._rsquare(4)
        self.assertFalse(np.isinf(out.to_numpy()).any(), f"rolling Rsquare leaked inf: {out.tolist()}")

    def test_well_conditioned_series_unchanged(self):
        # A clearly non-constant series must still produce a valid R^2 in [0, 1].
        op = Rsquare(_StubFeature(pd.Series([1.0, 2.0, 3.1, 3.9, 5.2, 6.0])), 0)
        out = op._load_internal("SH600000", None, None).dropna()
        self.assertFalse(np.isinf(out.to_numpy()).any())
        self.assertTrue(((out >= 0.0) & (out <= 1.0)).all(), f"R^2 outside [0, 1]: {out.tolist()}")


if __name__ == "__main__":
    unittest.main()
