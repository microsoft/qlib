import unittest
import numpy as np
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
