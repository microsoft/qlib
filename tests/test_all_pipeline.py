# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

import qlib
from qlib.config import REG_CN
from qlib.utils import drop_nan_by_y_index
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.estimator.handler import Alpha158
from qlib.contrib.strategy.strategy import TopkDropoutStrategy
from qlib.contrib.evaluate import (
    backtest as normal_backtest,
    risk_analysis,
)
from qlib.utils import exists_qlib_data


DATA_HANDLER_CONFIG = {
    "dropna_label": True,
    "start_date": "2008-01-01",
    "end_date": "2020-08-01",
    "market": "CSI300",
}

MODEL_CONFIG = {
    "loss": "mse",
    "colsample_bytree": 0.8879,
    "learning_rate": 0.0421,
    "subsample": 0.8789,
    "lambda_l1": 205.6999,
    "lambda_l2": 580.9768,
    "max_depth": 8,
    "num_leaves": 210,
    "num_threads": 20,
}

TRAINER_CONFIG = {
    "train_start_date": "2008-01-01",
    "train_end_date": "2014-12-31",
    "validate_start_date": "2015-01-01",
    "validate_end_date": "2016-12-31",
    "test_start_date": "2017-01-01",
    "test_end_date": "2020-08-01",
}

STRATEGY_CONFIG = {
    "topk": 50,
    "n_drop": 5,
}

BACKTEST_CONFIG = {
    "verbose": False,
    "limit_threshold": 0.095,
    "account": 100000000,
    "benchmark": "SH000300",
    "deal_price": "close",
    "open_cost": 0.0005,
    "close_cost": 0.0015,
    "min_cost": 5,
}


# train
def train():
    """train model

    Returns
    -------
        pred_score: pandas.DataFrame
            predict scores
        performance: dict
            model performance
    """
    # get data
    x_train, y_train, x_validate, y_validate, x_test, y_test = Alpha158(
        **DATA_HANDLER_CONFIG
    ).get_split_data(**TRAINER_CONFIG)

    # train
    model = LGBModel(**MODEL_CONFIG)
    model.fit(x_train, y_train, x_validate, y_validate)
    _pred = model.predict(x_test)
    _pred = pd.DataFrame(_pred, index=x_test.index, columns=y_test.columns)
    pred_score = pd.DataFrame(index=_pred.index)
    pred_score["score"] = _pred.iloc(axis=1)[0]

    # get performance
    try:
        model_score = model.score(x_test, y_test)
    except NotImplementedError:
        model_score = None
    # Remove rows from x, y and w, which contain Nan in any columns in y_test.
    x_test, y_test, __ = drop_nan_by_y_index(x_test, y_test)
    pred_test = model.predict(x_test)
    model_pearsonr = pearsonr(np.ravel(pred_test), np.ravel(y_test.values))[0]

    return pred_score, {"model_score": model_score, "model_pearsonr": model_pearsonr}


def backtest(pred):
    """backtest

    Parameters
    ----------
    pred: pandas.DataFrame
        predict scores

    Returns
    -------
    report_normal: pandas.DataFrame

    positions_normal: dict

    """
    strategy = TopkDropoutStrategy(**STRATEGY_CONFIG)
    _report_normal, _positions_normal = normal_backtest(pred, strategy=strategy, **BACKTEST_CONFIG)
    return _report_normal, _positions_normal


def analyze(report_normal):
    _analysis = dict()
    _analysis["excess_return_without_cost"] = risk_analysis(report_normal["return"] - report_normal["bench"])
    _analysis["excess_return_with_cost"] = risk_analysis(report_normal["return"] - report_normal["bench"] - report_normal["cost"])
    analysis_df = pd.concat(_analysis)  # type: pd.DataFrame
    print(analysis_df)
    return analysis_df


class TestAllFlow(unittest.TestCase):
    PRED_SCORE = None
    REPORT_NORMAL = None
    POSITIONS = None

    @classmethod
    def setUpClass(cls) -> None:
        # use default data
        provider_uri = "~/.qlib/qlib_data/cn_data_simple"  # target_dir
        if not exists_qlib_data(provider_uri):
            print(f"Qlib data is not found in {provider_uri}")
            sys.path.append(str(Path(__file__).resolve().parent.parent.joinpath("scripts")))
            from get_data import GetData

            GetData().qlib_data_cn(name="qlib_data_cn_simple", target_dir=provider_uri)
        qlib.init(provider_uri=provider_uri, region=REG_CN)

    def test_0_train(self):
        TestAllFlow.PRED_SCORE, model_pearsonr = train()
        self.assertGreaterEqual(model_pearsonr["model_pearsonr"], 0, "train failed")

    def test_1_backtest(self):
        TestAllFlow.REPORT_NORMAL, TestAllFlow.POSITIONS = backtest(
            TestAllFlow.PRED_SCORE
        )
        analyze_df = analyze(TestAllFlow.REPORT_NORMAL)
        self.assertGreaterEqual(
            analyze_df.loc(axis=0)["excess_return_with_cost", "annualized_return"].values[0], 0.10, "backtest failed",
        )


def suite():
    _suite = unittest.TestSuite()
    _suite.addTest(TestAllFlow("test_0_train"))
    _suite.addTest(TestAllFlow("test_1_backtest"))
    return _suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
