# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import unittest
from qlib.backtest import backtest, order
from qlib.tests import TestAutoData
import pandas as pd
from pathlib import Path

DIRNAME = Path(__file__).absolute().resolve().parent


class FileStrTest(TestAutoData):

    TEST_INST = "SH600519"

    EXAMPLE_FILE = DIRNAME / "order_example.csv"

    DEAL_NUM_FOR_1000 = 123.47105436976445

    def _gen_orders(self) -> pd.DataFrame:
        headers = [
            "datetime",
            "instrument",
            "amount",
            "direction",
        ]
        orders = [
            # test cash limit for buying
            ["20200103", self.TEST_INST, "1000", "buy"],
            # test min_cost for buying
            ["20200103", self.TEST_INST, "1", "buy"],
            # test held stock limit for selling
            ["20200106", self.TEST_INST, "1000", "sell"],
            # test cash limit for buying
            ["20200107", self.TEST_INST, "1000", "buy"],
            # test min_cost for selling
            ["20200108", self.TEST_INST, "1", "sell"],
            # test selling all stocks
            ["20200110", self.TEST_INST, str(self.DEAL_NUM_FOR_1000), "sell"],
        ]
        return pd.DataFrame(orders, columns=headers).set_index(["datetime", "instrument"])

    def test_file_str(self):

        orders = self._gen_orders()
        print(orders)
        orders.to_csv(self.EXAMPLE_FILE)

        orders = pd.read_csv(self.EXAMPLE_FILE, index_col=["datetime", "instrument"])

        strategy_config = {
            "class": "FileOrderStrategy",
            "module_path": "qlib.contrib.strategy.rule_strategy",
            "kwargs": {"file": self.EXAMPLE_FILE},
        }

        freq = "day"
        start_time = "2020-01-01"
        end_time = "2020-01-16"
        codes = [self.TEST_INST]

        backtest_config = {
            "start_time": start_time,
            "end_time": end_time,
            "account": 30000,
            "benchmark": None,  # benchmark is not required here for trading
            "exchange_kwargs": {
                "freq": freq,
                "limit_threshold": 0.095,
                "deal_price": "close",
                "open_cost": 0.0005,
                "close_cost": 0.0015,
                "min_cost": 500,
                "codes": codes,
                "trade_unit": 2,
            },
            # "pos_type": "InfPosition"  # Position with infinitive position
        }
        executor_config = {
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {
                "time_per_step": freq,
                "generate_report": False,
                "verbose": True,
                "indicator_config": {
                    "show_indicator": False,
                },
            },
        }
        report_dict, indicator_dict = backtest(executor=executor_config, strategy=strategy_config, **backtest_config)

        # ffr valid
        ffr_dict = indicator_dict["1day"]["ffr"].to_dict()
        ffr_dict = {str(date).split()[0]: ffr_dict[date] for date in ffr_dict}
        assert ffr_dict["2020-01-03"] == 0
        assert ffr_dict["2020-01-06"] == self.DEAL_NUM_FOR_1000 / 1000
        assert ffr_dict["2020-01-07"] == self.DEAL_NUM_FOR_1000 / 1000
        assert ffr_dict["2020-01-08"] == 0
        assert ffr_dict["2020-01-10"] == 1

        self.EXAMPLE_FILE.unlink()


if __name__ == "__main__":
    unittest.main()
