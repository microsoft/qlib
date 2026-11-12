import unittest

import pandas as pd

from qlib.backtest.position import Position
from qlib.contrib.report.analysis_position.parse_position import parse_position


def _position(cash, holdings):
    return Position(
        cash=cash,
        position_dict={
            code: {"amount": amount, "price": price, "count": 1} for code, (amount, price) in holdings.items()
        },
    )


class ParsePositionTest(unittest.TestCase):
    def test_buy_hold_sell_status(self):
        d1, d2, d3, d4 = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"])
        positions = {
            d1: _position(100.0, {"A": (10, 1.0), "B": (20, 2.0)}),
            # A held, B sold, C bought
            d2: _position(200.0, {"A": (10, 1.5), "C": (5, 4.0)}),
            # no sells: A and C held, B bought back
            d3: _position(300.0, {"A": (10, 1.5), "B": (20, 2.5), "C": (5, 4.0)}),
            # A and C sold
            d4: _position(400.0, {"B": (20, 3.0)}),
        }
        df = parse_position(positions)

        self.assertEqual(df.index.names, ["instrument", "datetime"])
        self.assertEqual(list(df.columns), ["amount", "cash", "count", "price", "status", "weight"])
        status = df["status"].to_dict()
        self.assertEqual(
            status,
            {
                ("A", d1): 1,
                ("B", d1): 1,
                ("A", d2): 0,
                ("C", d2): 1,
                ("B", d2): -1,
                ("A", d3): 0,
                ("B", d3): 1,
                ("C", d3): 0,
                ("B", d4): 0,
                ("A", d4): -1,
                ("C", d4): -1,
            },
        )
        self.assertEqual(list(df.index), list(status))

        # a sold stock keeps the previous day's amount/price/weight but takes the current day's date and status
        self.assertEqual(df.loc[("B", d2), "amount"], 20)
        self.assertEqual(df.loc[("B", d2), "price"], 2.0)
        self.assertAlmostEqual(df.loc[("B", d2), "weight"], 40 / 50)
        self.assertEqual(df.loc[("B", d2), "cash"], 100.0)
        self.assertEqual(df.loc[("A", d4), "price"], 1.5)
        self.assertEqual(df.loc[("A", d2), "cash"], 200.0)

        # cash and account value are popped from the input positions
        for pos in positions.values():
            self.assertNotIn("cash", pos.position)
            self.assertNotIn("now_account_value", pos.position)


if __name__ == "__main__":
    unittest.main()
