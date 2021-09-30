#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.


import qlib
import fire
from qlib.config import REG_CN, HIGH_FREQ_CONFIG
from qlib.data import D
from qlib.utils import exists_qlib_data, init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
from qlib.tests.data import GetData
from qlib.backtest import collect_data


class NestedDecisionExecutionWorkflow:

    market = "csi300"
    benchmark = "SH000300"
    data_handler_config = {
        "start_time": "2008-01-01",
        "end_time": "2021-05-31",
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
                    "train": ("2007-01-01", "2014-12-31"),
                    "valid": ("2015-01-01", "2016-12-31"),
                    "test": ("2020-01-01", "2021-05-31"),
                },
            },
        },
    }

    port_analysis_config = {
        "executor": {
            "class": "NestedExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {
                "time_per_step": "day",
                "inner_executor": {
                    "class": "NestedExecutor",
                    "module_path": "qlib.backtest.executor",
                    "kwargs": {
                        "time_per_step": "30min",
                        "inner_executor": {
                            "class": "SimulatorExecutor",
                            "module_path": "qlib.backtest.executor",
                            "kwargs": {
                                "time_per_step": "5min",
                                "generate_portfolio_metrics": True,
                                "verbose": True,
                                "indicator_config": {
                                    "show_indicator": True,
                                },
                            },
                        },
                        "inner_strategy": {
                            "class": "TWAPStrategy",
                            "module_path": "qlib.contrib.strategy.rule_strategy",
                        },
                        "generate_portfolio_metrics": True,
                        "indicator_config": {
                            "show_indicator": True,
                        },
                    },
                },
                "inner_strategy": {
                    "class": "SBBStrategyEMA",
                    "module_path": "qlib.contrib.strategy.rule_strategy",
                    "kwargs": {
                        "instruments": market,
                        "freq": "1min",
                    },
                },
                "track_data": True,
                "generate_portfolio_metrics": True,
                "indicator_config": {
                    "show_indicator": True,
                },
            },
        },
        "backtest": {
            "start_time": "2020-09-20",
            "end_time": "2021-05-20",
            "account": 100000000,
            "exchange_kwargs": {
                "freq": "1min",
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
        provider_uri_day = "~/.qlib/qlib_data/cn_data"  # target_dir
        GetData().qlib_data(target_dir=provider_uri_day, region=REG_CN, version="v2", exists_skip=True)
        provider_uri_1min = HIGH_FREQ_CONFIG.get("provider_uri")
        GetData().qlib_data(
            target_dir=provider_uri_1min, interval="1min", region=REG_CN, version="v2", exists_skip=True
        )
        provider_uri_map = {"1min": provider_uri_1min, "day": provider_uri_day}
        qlib.init(provider_uri=provider_uri_map, dataset_cache=None, expression_cache=None)

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
        self.port_analysis_config["backtest"]["benchmark"] = self.benchmark

        with R.start(experiment_name="backtest"):

            recorder = R.get_recorder()
            par = PortAnaRecord(
                recorder,
                self.port_analysis_config,
                risk_analysis_freq=["day", "30min", "5min"],
                indicator_analysis_freq=["day", "30min", "5min"],
                indicator_analysis_method="value_weighted",
            )
            par.generate()

        # user could use following methods to analysis the position
        # report_normal_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
        # from qlib.contrib.report import analysis_position
        # analysis_position.report_graph(report_normal_df)

    def collect_data(self):
        self._init_qlib()
        model = init_instance_by_config(self.task["model"])
        dataset = init_instance_by_config(self.task["dataset"])
        self._train_model(model, dataset)
        executor_config = self.port_analysis_config["executor"]
        backtest_config = self.port_analysis_config["backtest"]
        backtest_config["benchmark"] = self.benchmark
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
    fire.Fire(NestedDecisionExecutionWorkflow)
