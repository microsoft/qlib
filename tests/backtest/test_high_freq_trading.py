from typing import List, Tuple, Union
from qlib.backtest.position import Position
from qlib.backtest import collect_data, format_decisions
from qlib.backtest.decision import BaseTradeDecision, TradeRangeByTime
import qlib
from qlib.tests import TestAutoData
import unittest
import pandas as pd


@unittest.skip("This test takes a lot of time due to the large size of high-frequency data")
class TestHFBacktest(TestAutoData):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass(enable_1min=True, enable_1d_type="full")

    def _gen_orders(self, inst, date, pos) -> pd.DataFrame:
        headers = [
            "datetime",
            "instrument",
            "amount",
            "direction",
        ]
        orders = [
            [date, inst, pos, "sell"],
        ]
        return pd.DataFrame(orders, columns=headers)

    def test_trading(self):
        # date = "2020-02-03"
        # inst = "SH600068"
        # pos = 2.0167
        pos = 100000
        inst, date = "SH600519", "2021-01-18"
        market = [inst]

        start_time = f"{date}"
        end_time = f"{date} 15:00"  # include the high-freq data on the end day
        freq_l0 = "day"
        freq_l1 = "30min"
        freq_l2 = "1min"

        orders = self._gen_orders(inst=inst, date=date, pos=pos * 0.90)

        strategy_config = {
            "class": "FileOrderStrategy",
            "module_path": "qlib.contrib.strategy.rule_strategy",
            "kwargs": {
                "trade_range": TradeRangeByTime("10:45", "14:44"),
                "file": orders,
            },
        }
        backtest_config = {
            "start_time": start_time,
            "end_time": end_time,
            "account": {
                "cash": 0,
                inst: pos,
            },
            "benchmark": None,  # benchmark is not required here for trading
            "exchange_kwargs": {
                "freq": freq_l2,  # use the most fine-grained data as the exchange
                "limit_threshold": 0.095,
                "deal_price": "close",
                "open_cost": 0.0005,
                "close_cost": 0.0015,
                "min_cost": 5,
                "codes": market,
                "trade_unit": 100,
            },
            # "pos_type": "InfPosition"  # Position with infinitive position
        }
        executor_config = {
            "class": "NestedExecutor",  # Level 1 Order execution
            "module_path": "qlib.backtest.executor",
            "kwargs": {
                "time_per_step": freq_l0,
                "inner_executor": {
                    "class": "NestedExecutor",  # Leve 2 Order Execution
                    "module_path": "qlib.backtest.executor",
                    "kwargs": {
                        "time_per_step": freq_l1,
                        "inner_executor": {
                            "class": "SimulatorExecutor",
                            "module_path": "qlib.backtest.executor",
                            "kwargs": {
                                "time_per_step": freq_l2,
                                "generate_portfolio_metrics": False,
                                "verbose": True,
                                "indicator_config": {
                                    "show_indicator": False,
                                },
                                "track_data": True,
                            },
                        },
                        "inner_strategy": {
                            "class": "TWAPStrategy",
                            "module_path": "qlib.contrib.strategy.rule_strategy",
                        },
                        "generate_portfolio_metrics": False,
                        "indicator_config": {
                            "show_indicator": True,
                        },
                        "track_data": True,
                    },
                },
                "inner_strategy": {
                    "class": "TWAPStrategy",
                    "module_path": "qlib.contrib.strategy.rule_strategy",
                },
                "generate_portfolio_metrics": False,
                "indicator_config": {
                    "show_indicator": True,
                },
                "track_data": True,
            },
        }

        ret_val = {}
        decisions = list(
            collect_data(executor=executor_config, strategy=strategy_config, **backtest_config, return_value=ret_val)
        )
        report, indicator = ret_val["report"], ret_val["indicator"]
        # NOTE: please refer to the docs of format_decisions
        # NOTE: `"track_data": True,`  is very NECESSARY for collecting the decision!!!!!
        f_dec = format_decisions(decisions)
        print(indicator["1day"][0])


if __name__ == "__main__":
    unittest.main()
