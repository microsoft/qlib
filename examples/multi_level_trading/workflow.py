#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.


import qlib
import fire
from qlib import backtest
from qlib.config import REG_CN, HIGH_FREQ_CONFIG
from qlib.data import D
from qlib.utils import exists_qlib_data, init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
from qlib.tests.data import GetData
from qlib.backtest import collect_data


class MultiLevelTradingWorkflow:

    market = "csi300"
    benchmark = "SH000300"

    data_handler_config = {
        "start_time": "2008-01-01",
        "end_time": "2021-01-20",
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
                    "test": ("2017-01-01", "2021-01-20"),
                },
            },
        },
    }

    port_analysis_config = {
        "executor": {
            "class": "NestedExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {
                "time_per_step": "week",
                "inner_executor": {
                    "class": "SimulatorExecutor",
                    "module_path": "qlib.backtest.executor",
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
                "generate_report": True,
                "track_data": True,
            },
        },
        "backtest": {
            "start_time": "2017-01-01",
            "end_time": "2020-08-01",
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

    def _init_qlib_with_backend(self):
        provider_uri_1min = HIGH_FREQ_CONFIG.get("provider_uri")
        if not exists_qlib_data(provider_uri_1min):
            print(f"Qlib data is not found in {provider_uri_1min}")
            GetData().qlib_data(target_dir=provider_uri_1min, interval="1min", region=REG_CN)

        # TODO: update new data
        provider_uri_day = "~/.qlib/qlib_data/cn_data"  # target_dir
        if not exists_qlib_data(provider_uri_day):
            print(f"Qlib data is not found in {provider_uri_day}")
            GetData().qlib_data(target_dir=provider_uri_day, region=REG_CN)

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
        qlib.init(provider_uri=provider_uri_day, **client_config)

    def _get_highfreq_config(self, model, dataset):

        executor_config = self.port_analysis_config["executor"]
        # update executor with hierarchical decison freq ["day", "1min"]
        executor_config["kwargs"]["time_per_step"] = "day"
        executor_config["kwargs"]["inner_executor"]["kwargs"]["time_per_step"] = "15min"
        backtest_config = self.port_analysis_config["backtest"]

        # yahoo highfreq data time
        backtest_config["start_time"] = "2020-09-20"
        backtest_config["end_time"] = "2021-01-20"

        # update benchmark, yahoo data don't have SH000300
        instruments = D.instruments(market="csi300")
        instrument_list = D.list_instruments(instruments=instruments, as_list=True)
        backtest_config["benchmark"] = instrument_list

        # update exchange config
        backtest_config["exchange_kwargs"]["freq"] = "1min"

        # set strategy
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

        return executor_config, strategy_config, backtest_config

    def backtest_highfreq(self):
        self._init_qlib_with_backend()
        model = init_instance_by_config(self.task["model"])
        dataset = init_instance_by_config(self.task["dataset"])
        self._train_model(model, dataset)
        executor_config, strategy_config, backtest_config = self._get_highfreq_config(model, dataset)

        highfreq_port_analysis_config = {
            "executor": executor_config,
            "strategy": strategy_config,
            "backtest": backtest_config,
        }

        with R.start(experiment_name="backtest_highfreq"):

            recorder = R.get_recorder()
            par = PortAnaRecord(recorder, highfreq_port_analysis_config, "day")
            par.generate()


if __name__ == "__main__":
    fire.Fire(MultiLevelTradingWorkflow)
