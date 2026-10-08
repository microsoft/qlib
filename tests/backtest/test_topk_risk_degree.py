# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Regression test for TopkDropoutStrategy honoring an overridden ``get_risk_degree()``.

``get_risk_degree()`` is documented as the dynamic market-timing hook
("Dynamically risk_degree will result in Market timing.").  A subclass that
overrides it (the documented way to implement market timing) must actually
affect how ``TopkDropoutStrategy`` sizes its buys.  This test gates all
exposure to cash by overriding ``get_risk_degree()`` to return ``0.0`` and
asserts the strategy never builds a stock position.
"""

import unittest

import pandas as pd

from qlib.data import D
from qlib.backtest import backtest
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
from qlib.tests import TestAutoData


class GateToCashTopk(TopkDropoutStrategy):
    """A TopkDropoutStrategy that gates gross exposure to cash via get_risk_degree().

    Returning 0.0 means "invest 0% of total value", i.e. hold only cash and buy
    nothing.  This mirrors the documented market-timing override.
    """

    def get_risk_degree(self, trade_step=None):
        return 0.0


class TopkRiskDegreeTest(TestAutoData):
    INSTS = ["SH600000", "SH600009", "SH600010", "SH600015", "SH600016"]
    START = "2020-01-02"
    END = "2020-01-16"
    ACCOUNT = 1_000_000

    def _make_signal(self) -> pd.Series:
        """A constant, distinct per-instrument score so topk selection is deterministic."""
        cal = D.calendar(start_time=self.START, end_time=self.END, freq="day")
        rows = []
        for rank, inst in enumerate(self.INSTS):
            score = float(len(self.INSTS) - rank)  # higher score for earlier instruments
            for dt in cal:
                rows.append((inst, dt, score))
        df = pd.DataFrame(rows, columns=["instrument", "datetime", "score"])
        return df.set_index(["instrument", "datetime"])["score"]

    def _run(self, strategy) -> pd.DataFrame:
        backtest_config = {
            "start_time": self.START,
            "end_time": self.END,
            "account": self.ACCOUNT,
            "benchmark": "SH000300",
            "exchange_kwargs": {
                "freq": "day",
                "limit_threshold": 0.095,
                "deal_price": "close",
                "open_cost": 0.0005,
                "close_cost": 0.0015,
                "min_cost": 5,
                "codes": self.INSTS,
                "trade_unit": 100,
            },
        }
        executor_config = {
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {
                "time_per_step": "day",
                "generate_portfolio_metrics": True,
                "verbose": False,
                "indicator_config": {"show_indicator": False},
            },
        }
        portfolio_dict, _ = backtest(executor=executor_config, strategy=strategy, **backtest_config)
        report_normal, _ = portfolio_dict["1day"]
        return report_normal

    def test_gate_to_cash_via_get_risk_degree(self):
        """A get_risk_degree() override returning 0.0 must result in zero stock holdings."""
        signal = self._make_signal()
        gated = GateToCashTopk(topk=3, n_drop=1, signal=signal)
        report = self._run(gated)
        # "value" is the total market value of stock holdings on each step.
        # Gated to cash, the strategy must never buy, so it stays 0 throughout.
        self.assertEqual(
            report["value"].abs().sum(),
            0.0,
            msg=(
                "TopkDropoutStrategy ignored the overridden get_risk_degree() and built a "
                f"stock position despite a 0.0 risk degree.\n{report[['value', 'cash']]}"
            ),
        )

    def test_base_strategy_does_buy(self):
        """Sanity check: the same setup with the default risk degree does build a position."""
        signal = self._make_signal()
        base = TopkDropoutStrategy(topk=3, n_drop=1, signal=signal)
        report = self._run(base)
        self.assertGreater(report["value"].abs().sum(), 0.0)


if __name__ == "__main__":
    unittest.main()
