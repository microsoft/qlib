# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
import math
import shutil
import unittest
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

import qlib
from qlib.config import C
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.config import REG_CN
from qlib.workflow.record_temp import SignalRecord, SigAnaRecord
from qlib.tests import TestAutoData
from qlib.portfolio.optimizer import EnhancedIndexingOptimizer
from qlib.model.riskmodel import StructuredCovEstimator
from qlib.data.dataset.loader import QlibDataLoader
from qlib.data.dataset.handler import DataHandler
from qlib.data import D
from qlib.utils import exists_qlib_data, init_instance_by_config

market = "all"
trade_gap = 21
label_config = "Ref($close, -{}) / Ref($close, -1) - 1".format(trade_gap)  # reconstruct portfolio once a month

provider_uri = "~/.qlib_ei/qlib_data/cn_data"  # target_dir
if not exists_qlib_data(provider_uri):
    print(f"Qlib data is not found in {provider_uri}")
    sys.path.append(str(Path.cwd().parent.joinpath("scripts")))
    from get_data import GetData
    GetData().qlib_data(target_dir=provider_uri, region=REG_CN)
qlib.init(provider_uri=provider_uri, region=REG_CN)

###################################
# train model
###################################
data_handler_config = {
    "start_time": "2008-01-01",
    "end_time": "2020-08-01",
    "fit_start_time": "2008-01-01",
    "fit_end_time": "2014-11-30",
    "instruments": market,
    "label": [label_config]
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
            "num_threads": 32,
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
                "train": ("2008-01-01", "2014-11-30"),
                "valid": ("2015-01-01", "2016-11-30"),
                "test": ("2017-01-01", "2018-01-01"),
            },
        },
    },
}


class CSI300:
    """Simulate CSI300 as the Benchmark for Enhanced Indexing to Track"""

    def __init__(self):
        # provider_uri = '/nfs_data/qlib_data/ycz_daily/qlib'
        # qlib.init(provider_uri=provider_uri, region=REG_CN, dataset_cache=None, expression_cache=None)
        self.csi_weight = D.features(D.instruments('csi300'), ['$csi300_weight'])

    def __call__(self, pd_index, trade_date):
        weights = np.zeros(len(pd_index))

        for idx, instrument in enumerate(pd_index):
            if (instrument, trade_date) in self.csi_weight.index:
                weight = self.csi_weight.loc[(instrument, trade_date)].values[0]
                if not math.isnan(weight):
                    weights[idx] = weight

        assert weights.sum() > 0, ' Fetch CSI Weights Error!'
        weights = weights / weights.sum()

        return weights


class EnhancedIndexingStrategy:
    """Enhanced Indexing Strategy"""

    def __init__(self):
        self.benchmark = CSI300()

        provider_uri = "~/.qlib_ei/qlib_data/cn_data"
        qlib.init(provider_uri=provider_uri, region=REG_CN)

        self.data_handler = DataHandler(market, "2015-01-01", "2019-01-01", QlibDataLoader(["$close"]))
        self.label_handler = DataHandler(market, "2015-01-01", "2019-01-01", QlibDataLoader([label_config]))
        self.cov_estimator = StructuredCovEstimator()
        self.optimizer = EnhancedIndexingOptimizer(lamb=0.1, delta=0.4, bench_dev=0.03, max_iters=50000)

    def update(self, score_series, current, pred_date):
        """
        Parameters
        -----------
        score_series : pd.Series
            stock_id , score.
        current : Position()
            current of account.
        trade_exchange : Exchange()
            exchange.
        trade_date : pd.Timestamp
            date.
        """
        print(score_series)
        score_series = score_series.dropna()

        # portfolio init weight
        init_weight = current.reindex(score_series.index, fill_value=0).values.squeeze()
        init_weight_sum = init_weight.sum()
        if init_weight_sum > 0:
            init_weight /= init_weight_sum

        # covariance estimation
        selector = (self.data_handler.get_range_selector(pred_date, 252), score_series.index)
        price = self.data_handler.fetch(selector, level=None, squeeze=True)
        F, cov_b, var_u = self.cov_estimator.predict(price, return_decomposed_components=True)

        # optimize target portfolio
        w_bench = self.benchmark(score_series.index, pred_date)
        passed_init_weight = init_weight if init_weight_sum > 0 else None
        # print(F)
        # print(cov_b)
        # print(var_u)
        # print(passed_init_weight)
        # print(w_bench)
        target_weight = self.optimizer(score_series.values, F, cov_b, var_u, passed_init_weight, w_bench)
        # print(target_weight)
        target = pd.DataFrame(data=target_weight, index=score_series.index)

        active_weights = target_weight - w_bench
        selector = (self.label_handler.get_range_selector(pred_date, 1), score_series.index)
        label = self.label_handler.fetch(selector, level=None, squeeze=True)
        alpha = 0
        for instrument, weight in zip(score_series.index, active_weights):
            delta = label.loc[(pred_date, instrument)]
            alpha += weight * (0 if math.isnan(delta) else delta)

        print(alpha)

        return alpha, target


def train():
    """train model

    Returns
    -------
        pred_score: pandas.DataFrame
            predict scores
        performance: dict
            model performance
    """

    # model initiation
    model = init_instance_by_config(task["model"])
    dataset = init_instance_by_config(task["dataset"])

    # start exp
    with R.start(experiment_name="workflow"):
        R.log_params(**flatten_dict(task))
        model.fit(dataset)

        # prediction
        recorder = R.get_recorder()
        rid = recorder.id
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()
        pred_score = sr.load()

        # calculate ic and ric
        sar = SigAnaRecord(recorder)
        sar.generate()
        ic = sar.load(sar.get_path("ic.pkl"))
        ric = sar.load(sar.get_path("ric.pkl"))

    return pred_score, {"ic": ic, "ric": ric}, rid


def backtest_analysis(scores):
    """backtest enhanced indexing

    Parameters
    ----------
        scores: pandas.DataFrame
                predict scores

    Returns
    -------
        sharpe_ratio: floating-point
            sharpe ratio of the enhanced indexing portfolio
    """

    # backtest and analysis
    with R.start(experiment_name="backtest_analysis"):
        strategy = EnhancedIndexingStrategy()
        dates = scores.index.get_level_values(0).unique()

        alphas = []
        current = pd.DataFrame()
        gap_between_next_trade = 0
        for date in tqdm(dates):
            if gap_between_next_trade == 0:
                score_series = scores.loc[date]
                alpha, current = strategy.update(score_series, current, date)
                alphas.append(alpha)
                gap_between_next_trade = trade_gap
            else:
                gap_between_next_trade -= 1

        alphas = np.array(alphas)
        sharpe_ratio = alphas.mean() / np.std(alphas)
        print('Sharpe:', sharpe_ratio)

        return sharpe_ratio


class TestAllFlow(TestAutoData):
    PRED_SCORE = None
    REPORT_NORMAL = None
    POSITIONS = None
    RID = None

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(str(Path(C["exp_manager"]["kwargs"]["uri"].strip("file:")).resolve()))

    def test_0_train(self):
        TestAllFlow.PRED_SCORE, ic_ric, TestAllFlow.RID = train()
        self.assertGreaterEqual(ic_ric["ic"].all(), 0, "train failed")
        self.assertGreaterEqual(ic_ric["ric"].all(), 0, "train failed")

    def test_1_backtest(self):
        sharpe_ratio = backtest_analysis(TestAllFlow.PRED_SCORE)
        self.assertGreaterEqual(
            sharpe_ratio,
            0.90,
            "backtest failed",
        )


def suite():
    _suite = unittest.TestSuite()
    _suite.addTest(TestAllFlow("test_0_train"))
    _suite.addTest(TestAllFlow("test_1_backtest"))
    return _suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
