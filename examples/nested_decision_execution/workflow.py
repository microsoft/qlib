#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.
"""
The expect result of `backtest` is following in current version

'The following are analysis results of benchmark return(1day).'
                       risk
mean               0.000651
std                0.012472
annualized_return  0.154967
information_ratio  0.805422
max_drawdown      -0.160445
'The following are analysis results of the excess return without cost(1day).'
                       risk
mean               0.001258
std                0.007575
annualized_return  0.299303
information_ratio  2.561219
max_drawdown      -0.068386
'The following are analysis results of the excess return with cost(1day).'
                       risk
mean               0.001110
std                0.007575
annualized_return  0.264280
information_ratio  2.261392
max_drawdown      -0.071842
[1706497:MainThread](2021-12-07 14:08:30,263) INFO - qlib.workflow - [record_temp.py:441] - Portfolio analysis record 'port_analysis_30minute.
pkl' has been saved as the artifact of the Experiment 2
'The following are analysis results of benchmark return(30minute).'
                       risk
mean               0.000078
std                0.003646
annualized_return  0.148787
information_ratio  0.935252
max_drawdown      -0.142830
('The following are analysis results of the excess return without '
 'cost(30minute).')
                       risk
mean               0.000174
std                0.003343
annualized_return  0.331867
information_ratio  2.275019
max_drawdown      -0.074752
'The following are analysis results of the excess return with cost(30minute).'
                       risk
mean               0.000155
std                0.003343
annualized_return  0.294536
information_ratio  2.018860
max_drawdown      -0.075579
[1706497:MainThread](2021-12-07 14:08:30,277) INFO - qlib.workflow - [record_temp.py:441] - Portfolio analysis record 'port_analysis_5minute.p
kl' has been saved as the artifact of the Experiment 2
'The following are analysis results of benchmark return(5minute).'
                       risk
mean               0.000015
std                0.001460
annualized_return  0.172170
information_ratio  1.103439
max_drawdown      -0.144807
'The following are analysis results of the excess return without cost(5minute).'
                       risk
mean               0.000028
std                0.001412
annualized_return  0.319771
information_ratio  2.119563
max_drawdown      -0.077426
'The following are analysis results of the excess return with cost(5minute).'
                       risk
mean               0.000025
std                0.001412
annualized_return  0.281536
information_ratio  1.866091
max_drawdown      -0.078194
[1706497:MainThread](2021-12-07 14:08:30,287) INFO - qlib.workflow - [record_temp.py:466] - Indicator analysis record 'indicator_analysis_1day
.pkl' has been saved as the artifact of the Experiment 2
'The following are analysis results of indicators(1day).'
        value
ffr  0.945821
pa   0.000324
pos  0.542882
[1706497:MainThread](2021-12-07 14:08:30,293) INFO - qlib.workflow - [record_temp.py:466] - Indicator analysis record 'indicator_analysis_30mi
nute.pkl' has been saved as the artifact of the Experiment 2
'The following are analysis results of indicators(30minute).'
        value
ffr  0.982910
pa   0.000037
pos  0.500806
[1706497:MainThread](2021-12-07 14:08:30,302) INFO - qlib.workflow - [record_temp.py:466] - Indicator analysis record 'indicator_analysis_5min
ute.pkl' has been saved as the artifact of the Experiment 2
'The following are analysis results of indicators(5minute).'
        value
ffr  0.991017
pa   0.000000
pos  0.000000
[1706497:MainThread](2021-12-07 14:08:30,627) INFO - qlib.timer - [log.py:113] - Time cost: 0.014s | waiting `async_log` Done
"""


from copy import deepcopy
import qlib
import fire
import pandas as pd
from qlib.constant import REG_CN
from qlib.config import HIGH_FREQ_CONFIG
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

    exp_name = "nested"

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
        with R.start(experiment_name=self.exp_name):
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
            "module_path": "qlib.contrib.strategy.signal_strategy",
            "kwargs": {
                "signal": (model, dataset),
                "topk": 50,
                "n_drop": 5,
            },
        }
        self.port_analysis_config["strategy"] = strategy_config
        self.port_analysis_config["backtest"]["benchmark"] = self.benchmark

        with R.start(experiment_name=self.exp_name, resume=True):
            recorder = R.get_recorder()
            par = PortAnaRecord(
                recorder,
                self.port_analysis_config,
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
            "module_path": "qlib.contrib.strategy.signal_strategy",
            "kwargs": {
                "signal": (model, dataset),
                "topk": 50,
                "n_drop": 5,
            },
        }
        data_generator = collect_data(executor=executor_config, strategy=strategy_config, **backtest_config)
        for trade_decision in data_generator:
            print(trade_decision)

    # the code below are for checking, users don't have to care about it
    # The tests can be categorized into 2 types
    # 1) comparing same backtest
    # - Basic test idea: the shared accumulated value are equal in multiple levels
    #   - Aligning the profit calculation between multiple levels and single levels.
    # 2) comparing different backtest
    # - Basic test idea:
    #   - the daily backtest will be similar as multi-level(the data quality makes this gap smaller)

    def check_diff_freq(self):
        self._init_qlib()
        exp = R.get_exp(experiment_name="backtest")
        rec = next(iter(exp.list_recorders().values()))  # assuming this will get the latest recorder
        for check_key in "account", "total_turnover", "total_cost":
            check_key = "total_cost"

            acc_dict = {}
            for freq in ["30minute", "5minute", "1day"]:
                acc_dict[freq] = rec.load_object(f"portfolio_analysis/report_normal_{freq}.pkl")[check_key]
            acc_df = pd.DataFrame(acc_dict)
            acc_resam = acc_df.resample("1d").last().dropna()
            assert (acc_resam["30minute"] == acc_resam["1day"]).all()

    def backtest_only_daily(self):
        """
        This backtest is used for comparing the nested execution and single layer execution
        Due to the low quality daily-level and miniute-level data, they are hardly comparable.
        So it is used for detecting serious bugs which make the results different greatly.

        .. code-block:: shell

            [1724971:MainThread](2021-12-07 16:24:31,156) INFO - qlib.workflow - [record_temp.py:441] - Portfolio analysis record 'port_analysis_1day.pkl'
            has been saved as the artifact of the Experiment 2
            'The following are analysis results of benchmark return(1day).'
                                   risk
            mean               0.000651
            std                0.012472
            annualized_return  0.154967
            information_ratio  0.805422
            max_drawdown      -0.160445
            'The following are analysis results of the excess return without cost(1day).'
                                   risk
            mean               0.001375
            std                0.006103
            annualized_return  0.327204
            information_ratio  3.475016
            max_drawdown      -0.024927
            'The following are analysis results of the excess return with cost(1day).'
                                   risk
            mean               0.001184
            std                0.006091
            annualized_return  0.281801
            information_ratio  2.998749
            max_drawdown      -0.029568
            [1724971:MainThread](2021-12-07 16:24:31,170) INFO - qlib.workflow - [record_temp.py:466] - Indicator analysis record 'indicator_analysis_1day.
            pkl' has been saved as the artifact of the Experiment 2
            'The following are analysis results of indicators(1day).'
                 value
            ffr    1.0
            pa     0.0
            pos    0.0
            [1724971:MainThread](2021-12-07 16:24:31,188) INFO - qlib.timer - [log.py:113] - Time cost: 0.007s | waiting `async_log` Done

        """
        self._init_qlib()
        model = init_instance_by_config(self.task["model"])
        dataset = init_instance_by_config(self.task["dataset"])
        self._train_model(model, dataset)
        strategy_config = {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy.signal_strategy",
            "kwargs": {
                "signal": (model, dataset),
                "topk": 50,
                "n_drop": 5,
            },
        }
        pa_conf = deepcopy(self.port_analysis_config)
        pa_conf["strategy"] = strategy_config
        pa_conf["executor"] = {
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {
                "time_per_step": "day",
                "generate_portfolio_metrics": True,
                "verbose": True,
            },
        }
        pa_conf["backtest"]["benchmark"] = self.benchmark

        with R.start(experiment_name=self.exp_name, resume=True):
            recorder = R.get_recorder()
            par = PortAnaRecord(recorder, pa_conf)
            par.generate()


if __name__ == "__main__":
    fire.Fire(NestedDecisionExecutionWorkflow)
