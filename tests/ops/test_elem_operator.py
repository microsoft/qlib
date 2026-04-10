import unittest
import numpy as np
import pandas as pd
import pytest

from qlib.data import DatasetProvider
from qlib.data.data import ExpressionD
from qlib.tests import TestOperatorData, TestMockData, MOCK_DF
from qlib.config import C


class TestElementOperator(TestMockData):
    def setUp(self) -> None:
        self.instrument = "0050"
        self.start_time = "2022-01-01"
        self.end_time = "2022-02-01"
        self.freq = "day"
        self.mock_df = MOCK_DF[MOCK_DF["symbol"] == self.instrument]

    def test_Abs(self):
        field = "Abs($close-Ref($close, 1))"
        result = ExpressionD.expression(self.instrument, field, self.start_time, self.end_time, self.freq)
        self.assertGreaterEqual(result.min(), 0)
        result = result.to_numpy()
        prev_close = self.mock_df["close"].shift(1)
        close = self.mock_df["close"]
        change = prev_close - close
        golden = change.abs().to_numpy()
        self.assertIsNone(np.testing.assert_allclose(result, golden))

    def test_Sign(self):
        field = "Sign($close-Ref($close, 1))"
        result = ExpressionD.expression(self.instrument, field, self.start_time, self.end_time, self.freq)
        result = result.to_numpy()
        prev_close = self.mock_df["close"].shift(1)
        close = self.mock_df["close"]
        change = close - prev_close
        change[change > 0] = 1.0
        change[change < 0] = -1.0
        golden = change.to_numpy()
        self.assertIsNone(np.testing.assert_allclose(result, golden))

    def test_WMA(self):
        # Regression test for issue #1993: WMA.weighted_mean divided by the
        # window length twice (normalized weights, then np.nanmean) producing
        # values ~N times smaller than a correct weighted moving average.
        # The sister operator EMA.exp_weighted_mean correctly uses np.nansum.
        N = 5
        field = f"WMA($close, {N})"
        result = ExpressionD.expression(self.instrument, field, self.start_time, self.end_time, self.freq)
        result = result.to_numpy()

        close = self.mock_df["close"].reset_index(drop=True).to_numpy()

        def reference_weighted_mean(x):
            w = np.arange(len(x)) + 1
            w = w / w.sum()
            return np.nansum(w * x)

        golden = (
            pd.Series(close)
            .rolling(N, min_periods=1)
            .apply(reference_weighted_mean, raw=True)
            .to_numpy()
        )
        np.testing.assert_allclose(result, golden, rtol=1e-5)

        # A weighted moving average with non-negative weights summing to one
        # must stay inside the input series' range. The pre-fix implementation
        # produced values ~close/N, far below close.min().
        close_min = float(np.nanmin(close))
        close_max = float(np.nanmax(close))
        self.assertGreaterEqual(float(np.nanmin(result)), close_min - 1e-9)
        self.assertLessEqual(float(np.nanmax(result)), close_max + 1e-9)


class TestOperatorDataSetting(TestOperatorData):
    def test_setting(self):
        self.assertEqual(len(self.instruments_d), 1)
        self.assertGreater(len(self.cal), 0)


class TestInstElementOperator(TestOperatorData):
    def setUp(self) -> None:
        freq = "day"
        expressions = [
            "$change",
            "Abs($change)",
        ]
        columns = ["change", "abs"]
        self.data = DatasetProvider.inst_calculator(
            self.inst, self.start_time, self.end_time, freq, expressions, self.spans, C, []
        )
        self.data.columns = columns

    @pytest.mark.slow
    def test_abs(self):
        abs_values = self.data["abs"]
        self.assertGreater(abs_values[2], 0)


if __name__ == "__main__":
    unittest.main()
