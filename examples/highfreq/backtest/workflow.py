#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import sys
from pathlib import Path

import qlib
import pandas as pd
from qlib.config import REG_CN
from qlib.contrib.strategy import TopkDropoutStrategy
from qlib.contrib.backtest import backtest
from qlib.utils import exists_qlib_data, init_instance_by_config, flatten_dict
from qlib.tests.data import GetData

if __name__ == "__main__":

    # use default data
    provider_uri = "~/.qlib/qlib_data/cn_data"  # target_dir
    if not exists_qlib_data(provider_uri):
        print(f"Qlib data is not found in {provider_uri}")
        GetData().qlib_data(target_dir=provider_uri, region=REG_CN)

    qlib.init(provider_uri=provider_uri, region=REG_CN)

    market = "csi300"
    benchmark = "SH000300"

    ###################################
    # train model
    ###################################
    
    data_handler_config = {
        "start_time": "2008-01-01",
        "end_time": "2020-08-01",
        "fit_start_time": "2008-01-01",
        "fit_end_time": "2014-12-31",
        "instruments": market,
    }

    task = {
        "model": {
            "class": "LGBModel",
            "module_path": "qlib.contrib.model.gbdt",
            "kwargs": {
                "loss": "mse",
                "colsample_bytree": 0.8879,
                "learning_rate": 0.0421,
                "subsample": 0.8789,
                "lambda_l1": 205.6999,
                "lambda_l2": 580.9768,
                "max_depth": 8,
                "num_leaves": 210,
                "num_threads": 20,
            },
        },
        "dataset": {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": {
                    "class": "Alpha158",
                    "module_path": "qlib.contrib.data.handler",
                    "kwargs": data_handler_config,
                },
                "segments": {
                    "train": ("2012-01-01", "2014-12-31"),
                    "valid": ("2015-01-01", "2016-12-31"),
                    "test": ("2017-01-01", "2018-01-31"),
                },
            },
        },
    }
        # model initialization
    model = init_instance_by_config(task["model"])
    dataset = init_instance_by_config(task["dataset"])
    model.fit(dataset)

    trade_start_time = "2017-01-31"
    trade_end_time = "2018-01-31"

    backtest_config={
        "strategy": {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy.model_strategy",
            "kwargs": {
                "step_bar": "week",
                "model": model,
                "dataset": dataset,
                "topk": 50,
                "n_drop": 5,
            },
        },
        "env":{
            "class": "SplitEnv",
            "module_path": "qlib.contrib.backtest.env",
            "kwargs": {
                "step_bar": "week",
                "sub_env": {
                    "class": "SimulatorEnv",
                    "module_path": "qlib.contrib.backtest.env",
                    "kwargs": {
                        "step_bar": "day",
                        "verbose": True,
                    }
                },
                "sub_strategy": {
                    "class": "SBBStrategyEMA",
                    "module_path": "qlib.contrib.strategy.rule_strategy",
                    "kwargs": {
                        "step_bar": "day",
                        "freq": "day",
                        "instruments": "csi300",
                    }
                }
            }
        },
        "backtest":{
            "start_time": trade_start_time,
            "end_time": trade_end_time,
            "verbose": False,
            "limit_threshold": 0.095,
            "account": 100000000,
            "benchmark": benchmark,
            "deal_price": "close",
            "open_cost": 0.0005,
            "close_cost": 0.0015,
            "min_cost": 5,
        }
    }


    report_dict = backtest(start_time=trade_start_time, end_time=trade_end_time, **backtest_config, account=1e8, deal_price="$close", verbose=False)