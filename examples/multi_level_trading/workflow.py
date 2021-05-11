#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import sys
from pathlib import Path

import qlib
import pandas as pd
from qlib.config import REG_CN

from qlib.utils import exists_qlib_data, init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
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
                    "train": ("2008-01-01", "2014-12-31"),
                    "valid": ("2015-01-01", "2016-12-31"),
                    "test": ("2017-01-01", "2020-08-01"),
                },
            },
        },
    }
    # model initialization
    model = init_instance_by_config(task["model"])
    dataset = init_instance_by_config(task["dataset"])

    trade_start_time = "2017-01-01"
    trade_end_time = "2020-08-01"

    port_analysis_config = {
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
        "env": {
            "class": "SplitExecutor",
            "module_path": "qlib.contrib.backtest.executor",
            "kwargs": {
                "step_bar": "week",
                "sub_env": {
                    "class": "SimulatorExecutor",
                    "module_path": "qlib.contrib.backtest.executor",
                    "kwargs": {
                        "step_bar": "day",
                        "verbose": True,
                        "generate_report": True,
                    },
                },
                "sub_strategy": {
                    "class": "SBBStrategyEMA",
                    "module_path": "qlib.contrib.strategy.rule_strategy",
                    "kwargs": {
                        "step_bar": "day",
                        "freq": "day",
                        "instruments": market,
                    },
                },
            },
        },
        "backtest": {
            "start_time": trade_start_time,
            "end_time": trade_end_time,
            "account": 100000000,
            "benchmark": benchmark,
            "exchange_kwargs": {
                "freq": "day",
                "verbose": False,
                "limit_threshold": 0.095,
                "deal_price": "close",
                "open_cost": 0.0005,
                "close_cost": 0.0015,
                "min_cost": 5,
            },
        },
    }

    with R.start(experiment_name="highfreq_backtest"):
        R.log_params(**flatten_dict(task))
        model.fit(dataset)
        R.save_objects(**{"params.pkl": model})

        # prediction
        recorder = R.get_recorder()
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()

        # backtest. If users want to use backtest based on their own prediction,
        # please refer to https://qlib.readthedocs.io/en/latest/component/recorder.html#record-template.
        par = PortAnaRecord(recorder, port_analysis_config, "day")
        par.generate()
