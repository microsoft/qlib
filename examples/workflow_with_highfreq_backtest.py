#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import sys
from pathlib import Path

import qlib
import pandas as pd
from qlib.config import REG_CN
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.data.handler import Alpha158
from qlib.contrib.strategy.strategy import TopkDropoutStrategy
from qlib.contrib.evaluate import (
    backtest as normal_backtest,
    risk_analysis,
)
from qlib.utils import exists_qlib_data, init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord


if __name__ == "__main__":

    # use default data
    provider_uri = "~/.qlib/qlib_data/cn_data"  # target_dir
    if not exists_qlib_data(provider_uri):
        print(f"Qlib data is not found in {provider_uri}")
        sys.path.append(str(Path(__file__).resolve().parent.parent.joinpath("scripts")))
        from get_data import GetData

        GetData().qlib_data(target_dir=provider_uri, region=REG_CN)

    qlib.init(provider_uri=provider_uri, region=REG_CN)

    market = "csi300"
    benchmark = "SH000300"

    ###################################
    # train model
    ###################################
    data_handler_config = {
        "start_time": "2012-01-01",
        "end_time": "2019-06-01",
        "fit_start_time": "2012-01-01",
        "fit_end_time": "2017-04-30",
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
                    "train": ("2012-01-01", "2017-04-30"),
                    "valid": ("2017-05-01", "2019-04-30"),
                    "test": ("2019-05-01", "2019-06-01"),
                },
            },
        },
    }

    highfreq_executor_config = {
        "log_dir": '/shared_data/data/v-xiabi/highfreq-exe/log/',
        "is_multi": True,
        "resources": {
            "num_cpus": 48,
            "num_gpus": 2,
            'device': 'cpu',
        },
        "paths": {
            "raw_dir": "/shared_data/data/v-xiabi/highfreq-exe/data/backtest_test_multi",
            "feature_conf": "/shared_data/data/v-xiabi/highfreq-exe/code/rl4execution/config/test_feature_all1620.json",
        },
        "env_conf": {
            "name": "MARL_Accelerated",
            "max_step_num": 237,
            "limit": 10,
            "time_interval": 30,
            "interval_num": 8,
            "features": "raw_30",
            "max_agent_num": 49,
            "log": True,
            "obs": {
                "name": "MultiTeacherObs",
                "config": {}
            },
            "action": {
                "name": "Multi_Static",
                "config": {
                    'action_num':5,
                    'action_map': [0, 0.25, 0.5, 0.75, 1],
                }
            },
            "reward": {
                "name": "Multi_VP_Penalty_small",
                "config": {
                    "action_penalty": 100,
                    "hit_penalty": 1.,
                }
            },
        },
        "policy_conf": {
            "name": "Multi_RL_backtest",
            "config": {
                "buy_policy": '/shared_data/data/v-xiabi/highfreq-exe/model/OPDS_buy/policy_best',
                'sell_policy': '/shared_data/data/v-xiabi/highfreq-exe/model/OPDS_sell/policy_best',
            },
        },
    }
    
    port_analysis_config = {
        "strategy": {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy.strategy",
            "kwargs": {
                "topk": 50,
                "n_drop": 5,
            },
        },
        "backtest": {
            "verbose": False,
            "limit_threshold": 0.095,
            "account": 100000000,
            "benchmark": benchmark,
            "deal_price": "close",
            "open_cost": 0.0005,
            "close_cost": 0.0015,
            "min_cost": 5,
            "highfreq_executor": {
                "class": "Online_Executor",
                "module_path": "/shared_data/data/v-xiabi/highfreq-exe/code/rl4execution/executor.py",
                "kwargs": highfreq_executor_config,
            }
        },
    }

    # model initiaiton
    model = init_instance_by_config(task["model"])
    dataset = init_instance_by_config(task["dataset"])

    # start exp
    with R.start(experiment_name="workflow"):
        R.log_params(**flatten_dict(task))
        model.fit(dataset)

        # prediction
        recorder = R.get_recorder()
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()

        # backtest
        par = PortAnaRecord(recorder, port_analysis_config)
        par.generate()
