# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import unittest

from qlib.backtest.exchange import Exchange


def make_exchange(min_cost: float) -> Exchange:
    """Build an Exchange for testing `_get_buy_amount_by_cash_limit` only.

    The method is pure arithmetic over `min_cost`, so the instance is created
    without `__init__` to keep the test free of the data layer.
    """
    exchange = object.__new__(Exchange)
    exchange.min_cost = min_cost
    return exchange


class TestBuyAmountByCashLimit(unittest.TestCase):
    """`_get_buy_amount_by_cash_limit` must handle a zero proportional fee."""

    TRADE_PRICE = 10.0
    CASH = 1000.0

    def test_default_cost_ratio(self):
        """With the default fees, the min_cost branch applies at this cash level."""
        exchange = make_exchange(min_cost=5.0)
        # critical_price = 5 / 0.0015 + 5 = 3338.3 > cash, so the fee is min_cost.
        amount = exchange._get_buy_amount_by_cash_limit(self.TRADE_PRICE, self.CASH, cost_ratio=0.0015)
        self.assertAlmostEqual(amount, (self.CASH - 5.0) / self.TRADE_PRICE)

    def test_cost_ratio_above_critical_price(self):
        """Above the critical price the proportional fee applies."""
        exchange = make_exchange(min_cost=5.0)
        cash = 10_000.0
        amount = exchange._get_buy_amount_by_cash_limit(self.TRADE_PRICE, cash, cost_ratio=0.0015)
        self.assertAlmostEqual(amount, cash / 1.0015 / self.TRADE_PRICE)

    def test_zero_cost_ratio_with_min_cost(self):
        """A zero proportional fee must not divide by zero; min_cost still applies."""
        exchange = make_exchange(min_cost=5.0)
        amount = exchange._get_buy_amount_by_cash_limit(self.TRADE_PRICE, self.CASH, cost_ratio=0.0)
        self.assertAlmostEqual(amount, (self.CASH - 5.0) / self.TRADE_PRICE)

    def test_frictionless(self):
        """With no fee at all the whole cash balance is investable."""
        exchange = make_exchange(min_cost=0.0)
        amount = exchange._get_buy_amount_by_cash_limit(self.TRADE_PRICE, self.CASH, cost_ratio=0.0)
        self.assertAlmostEqual(amount, self.CASH / self.TRADE_PRICE)

    def test_cash_below_min_cost(self):
        """Cash that cannot even cover the minimum fee buys nothing."""
        exchange = make_exchange(min_cost=5.0)
        amount = exchange._get_buy_amount_by_cash_limit(self.TRADE_PRICE, 1.0, cost_ratio=0.0)
        self.assertEqual(amount, 0.0)


if __name__ == "__main__":
    unittest.main()
