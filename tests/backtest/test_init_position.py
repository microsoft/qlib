# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import unittest
import qlib
from qlib.backtest import backtest, order
from qlib.tests import TestAutoData
from qlib.backtest.order import TradeDecisionWO, TradeRangeByTime
import pandas as pd
from pathlib import Path


class FileStrTest(TestAutoData):

    TEST_INST = "SH600519"

    def init_qlib(self):
        provider_uri_day = "/nfs_data1/stock_data/huaxia_1d_qlib"
        provider_uri_1min = "/nfs_data1/stock_data/huaxia_1min_qlib"
        provider_uri_map = {"1min": provider_uri_1min, "day": provider_uri_day}

        client_config = {
            "calendar_provider": {
                "class": "LocalCalendarProvider",
                "module_path": "qlib.data.data",
                "kwargs": {
                    "backend": {
                        "class": "FileCalendarStorage",
                        "module_path": "qlib.data.storage.file_storage",
                        "kwargs": {"provider_uri_map": provider_uri_map},
                    }
                },
            },
            "feature_provider": {
                "class": "LocalFeatureProvider",
                "module_path": "qlib.data.data",
                "kwargs": {
                    "backend": {
                        "class": "FileFeatureStorage",
                        "module_path": "qlib.data.storage.file_storage",
                        "kwargs": {"provider_uri_map": provider_uri_map},
                    }
                },
            },
        }
        qlib.init(provider_uri=provider_uri_day, **client_config, expression_cache=None, dataset_cache=None)

    def test_file_str(self):
        freq = "1min"
        inst = ["SH600000", "SH600011"]
        start_time = "2020-01-01"
        end_time = "2020-01-15 15:00"

        strategy_config = {
            "class": "RandomOrderStrategy",
            "module_path": "qlib.contrib.strategy.rule_strategy",
            "kwargs": {
                "trade_range": TradeRangeByTime("9:30", "15:00"),
                "sample_ratio": 1.0,
                "volume_ratio": 0.01,
                "market": inst,
            },
        }
        position_dict = {
            "cash": 100000000,
            "SH600000": {"amount": 100},
            "SH600011": {"amount": 101},
        }
        backtest_config = {
            "start_time": start_time,
            "end_time": end_time,
            "account": position_dict,
            "benchmark": None,  # benchmark is not required here for trading
            "exchange_kwargs": {
                "freq": freq,
                "limit_threshold": 0.095,
                "deal_price": "close",
                "open_cost": 0.0005,
                "close_cost": 0.0015,
                "min_cost": 5,
                "codes": inst,
            },
            "pos_type": "Position",  # Position with infinitive position
        }
        executor_config = {
            "class": "NestedExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {
                "time_per_step": "day",
                "inner_executor": {
                    "class": "SimulatorExecutor",
                    "module_path": "qlib.backtest.executor",
                    "kwargs": {
                        "time_per_step": freq,
                        "generate_report": False,
                        "verbose": False,
                        # "verbose": True,
                        "indicator_config": {
                            "show_indicator": False,
                        },
                    },
                },
                "inner_strategy": {
                    "class": "TWAPStrategy",
                    "module_path": "qlib.contrib.strategy.rule_strategy",
                },
                "track_data": True,
                "generate_report": True,
                "indicator_config": {
                    "show_indicator": True,
                },
            },
        }
        self.init_qlib()
        backtest(executor=executor_config, strategy=strategy_config, **backtest_config)


if __name__ == "__main__":
    unittest.main()
