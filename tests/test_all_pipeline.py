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
from qlib.contrib.estimator.handler import QLibDataHandlerV1
from qlib.contrib.strategy.strategy import TopkAmountStrategy
from qlib.contrib.evaluate import (
    backtest as normal_backtest,
    long_short_backtest,
    risk_analysis,
)
from qlib.utils import exists_qlib_data


DATA_HANDLER_CONFIG = {
    "dropna_label": True,
    "start_date": "2007-01-01",
    "end_date": "2020-08-01",
    "market": "CSI500",
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
    "train_start_date": "2007-01-01",
    "train_end_date": "2014-12-31",
    "validate_start_date": "2015-01-01",
    "validate_end_date": "2016-12-31",
    "test_start_date": "2017-01-01",
    "test_end_date": "2020-08-01",
}

STRATEGY_CONFIG = {
    "topk": 50,
    "buffer_margin": 230,
}

BACKTEST_CONFIG = {
    "verbose": False,
    "limit_threshold": 0.095,
    "account": 100000000,
    "benchmark": "SH000905",
    "deal_price": "vwap",
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
    x_train, y_train, x_validate, y_validate, x_test, y_test = QLibDataHandlerV1(**DATA_HANDLER_CONFIG).get_split_data(
        **TRAINER_CONFIG
    )

    # train
    model = LGBModel(**MODEL_CONFIG)
    model.fit(x_train, y_train, x_validate, y_validate)
    _pred = model.predict(x_test)
    _pred = pd.DataFrame(_pred, index=x_test.index, columns=y_test.columns)
    pred_score = pd.DataFrame(index=_pred.index)
    pred_score["score"] = _pred.iloc(axis=1)[0]

    # get performance
    model_score = model.score(x_test, y_test)
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

    long_short_reports: dict

    """
    strategy = TopkAmountStrategy(**STRATEGY_CONFIG)
    _report_normal, _positions_normal = normal_backtest(pred, strategy=strategy, **BACKTEST_CONFIG)
    _long_short_reports = long_short_backtest(pred, topk=50)
    return _report_normal, _positions_normal, _long_short_reports


def analyze(report_normal, long_short_reports):
    _analysis = dict()
    _analysis["pred_long"] = risk_analysis(long_short_reports["long"])
    _analysis["pred_short"] = risk_analysis(long_short_reports["short"])
    _analysis["pred_long_short"] = risk_analysis(long_short_reports["long_short"])
    _analysis["sub_bench"] = risk_analysis(report_normal["return"] - report_normal["bench"])
    _analysis["sub_cost"] = risk_analysis(report_normal["return"] - report_normal["bench"] - report_normal["cost"])
    analysis_df = pd.concat(_analysis)  # type: pd.DataFrame
    print(analysis_df)
    return analysis_df


class TestAllFlow(unittest.TestCase):
    PRED_SCORE = None
    REPORT_NORMAL = None
    POSITIONS = None
    LONG_SHORT_REPORTS = None

    @classmethod
    def setUpClass(cls) -> None:
        # use default data
        mount_path = "~/.qlib/qlib_data/cn_data"  # target_dir
        if not exists_qlib_data(mount_path):
            print(f"Qlib data is not found in {mount_path}")
            sys.path.append(str(Path(__file__).resolve().parent.parent.joinpath("scripts")))
            from get_data import GetData

            GetData().qlib_data_cn(mount_path)
        qlib.init(mount_path=mount_path, region=REG_CN)

    def test_0_train(self):
        TestAllFlow.PRED_SCORE, model_pearsonr = train()
        self.assertGreaterEqual(model_pearsonr["model_pearsonr"], 0, "train failed")

    def test_1_backtest(self):
        TestAllFlow.REPORT_NORMAL, TestAllFlow.POSITIONS, TestAllFlow.LONG_SHORT_REPORTS = backtest(
            TestAllFlow.PRED_SCORE
        )
        analyze_df = analyze(TestAllFlow.REPORT_NORMAL, TestAllFlow.LONG_SHORT_REPORTS)
        self.assertGreaterEqual(
            analyze_df.loc(axis=0)["sub_cost", "annual"].values[0],
            0.10,
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
