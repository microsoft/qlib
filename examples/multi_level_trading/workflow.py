#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.


import qlib
import fire
from qlib.config import REG_CN

from qlib.utils import exists_qlib_data, init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
from qlib.tests.data import GetData
from qlib.contrib.backtest import collect_data


class MultiLevelTradingWorkflow:

    market = "csi300"
    benchmark = "SH000300"

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

    trade_start_time = "2017-01-01"
    trade_end_time = "2017-02-01"

    port_analysis_config = {
        "executor": {
            "class": "SplitExecutor",
            "module_path": "qlib.contrib.backtest.executor",
            "kwargs": {
                "time_per_step": "week",
                "inner_executor": {
                    "class": "SimulatorExecutor",
                    "module_path": "qlib.contrib.backtest.executor",
                    "kwargs": {
                        "time_per_step": "day",
                        "verbose": True,
                        "generate_report": True,
                    },
                },
                "inner_strategy": {
                    "class": "SBBStrategyEMA",
                    "module_path": "qlib.contrib.strategy.rule_strategy",
                    "kwargs": {
                        "freq": "day",
                        "instruments": market,
                    },
                },
                "track_data": True,
            },
        },
        "backtest": {
            "start_time": trade_start_time,
            "end_time": trade_end_time,
            "account": 100000000,
            "benchmark": benchmark,
            "exchange_kwargs": {
                "freq": "day",
                "limit_threshold": 0.095,
                "deal_price": "close",
                "open_cost": 0.0005,
                "close_cost": 0.0015,
                "min_cost": 5,
            },
        },
    }

    def _init_qlib(self):
        """initialize qlib"""
        provider_uri = "~/.qlib/qlib_data/cn_data"  # target_dir
        if not exists_qlib_data(provider_uri):
            print(f"Qlib data is not found in {provider_uri}")
            GetData().qlib_data(target_dir=provider_uri, region=REG_CN)
        qlib.init(provider_uri=provider_uri, region=REG_CN)

    def _train_model(self, model, dataset):
        with R.start(experiment_name="train"):
            R.log_params(**flatten_dict(self.task))
            model.fit(dataset)
            R.save_objects(**{"params.pkl": model})

            # prediction
            recorder = R.get_recorder()
            sr = SignalRecord(model, dataset, recorder)
            sr.generate()

    def backtest(self):
        self._init_qlib()
        model = init_instance_by_config(self.task["model"])
        dataset = init_instance_by_config(self.task["dataset"])
        self._train_model(model, dataset)
        strategy_config = {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy.model_strategy",
            "kwargs": {
                "model": model,
                "dataset": dataset,
                "topk": 50,
                "n_drop": 5,
            },
        }
        self.port_analysis_config["strategy"] = strategy_config
        with R.start(experiment_name="backtest"):

            recorder = R.get_recorder()
            par = PortAnaRecord(recorder, self.port_analysis_config, "day")
            par.generate()

    def collect_data(self):
        self._init_qlib()
        model = init_instance_by_config(self.task["model"])
        dataset = init_instance_by_config(self.task["dataset"])
        self._train_model(model, dataset)
        executor_config = self.port_analysis_config["executor"]
        backtest_config = self.port_analysis_config["backtest"]
        strategy_config = {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy.model_strategy",
            "kwargs": {
                "model": model,
                "dataset": dataset,
                "topk": 50,
                "n_drop": 5,
            },
        }
        data_generator = collect_data(executor=executor_config, strategy=strategy_config, **backtest_config)
        for trade_decision in data_generator:
            print(trade_decision)


if __name__ == "__main__":
    fire.Fire(MultiLevelTradingWorkflow)
