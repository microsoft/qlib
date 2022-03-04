import unittest

from qlib.data import DatasetProvider
from qlib.tests import TestOperatorData
from qlib.config import C


class TestOperatorDataSetting(TestOperatorData):
    def test_setting(self):
        self.assertEqual(len(self.instruments_d), 1)
        self.assertGreater(len(self.cal), 0)


class TestElementOperator(TestOperatorData):
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

    def test_abs(self):
        abs_values = self.data["abs"]
        self.assertGreater(abs_values[2], 0)


if __name__ == "__main__":
    unittest.main()
