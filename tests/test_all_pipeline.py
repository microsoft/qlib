# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
import shutil
import unittest
import pytest
from pathlib import Path

import qlib
from qlib.config import C
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, SigAnaRecord, PortAnaRecord
from qlib.tests import TestAutoData
from qlib.tests.config import CSI300_GBDT_TASK, CSI300_BENCH


def train(uri_path: str = None):
    """train model

    Returns
    -------
        pred_score: pandas.DataFrame
            predict scores
        performance: dict
            model performance
    """

    # model initialization
    model = init_instance_by_config(CSI300_GBDT_TASK["model"])
    dataset = init_instance_by_config(CSI300_GBDT_TASK["dataset"])
    # To test __repr__
    print(dataset)
    print(R)

    # start exp
    with R.start(experiment_name="workflow", uri=uri_path):
        R.log_params(**flatten_dict(CSI300_GBDT_TASK))
        model.fit(dataset)
        R.save_objects(trained_model=model)
        # prediction
        recorder = R.get_recorder()
        # To test __repr__
        print(recorder)
        # To test get_local_dir
        print(recorder.get_local_dir())
        rid = recorder.id
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()
        pred_score = sr.load("pred.pkl")

        # calculate ic and ric
        sar = SigAnaRecord(recorder)
        sar.generate()
        ic = sar.load("ic.pkl")
        ric = sar.load("ric.pkl")

        uri_path = R.get_uri()
    return pred_score, {"ic": ic, "ric": ric}, rid, uri_path


def fake_experiment():
    """A fake experiment workflow to test uri

    Returns
    -------
        pass_or_not_for_default_uri: bool
        pass_or_not_for_current_uri: bool
        temporary_exp_dir: str
    """

    # start exp
    default_uri = R.get_uri()
    current_uri = "file:./temp-test-exp-mag"
    with R.start(experiment_name="fake_workflow_for_expm", uri=current_uri):
        R.log_params(**flatten_dict(CSI300_GBDT_TASK))

        current_uri_to_check = R.get_uri()
    default_uri_to_check = R.get_uri()
    return default_uri == default_uri_to_check, current_uri == current_uri_to_check, current_uri


def backtest_analysis(pred, rid, uri_path: str = None):
    """backtest and analysis

    Parameters
    ----------
    rid : str
        the id of the recorder to be used in this function
    uri_path: str
        mlflow uri path

    Returns
    -------
    analysis : pandas.DataFrame
        the analysis result

    """
    with R.uri_context(uri=uri_path):
        recorder = R.get_recorder(experiment_name="workflow", recorder_id=rid)

    dataset = init_instance_by_config(CSI300_GBDT_TASK["dataset"])
    model = recorder.load_object("trained_model")

    port_analysis_config = {
        "executor": {
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {
                "time_per_step": "day",
                "generate_portfolio_metrics": True,
            },
        },
        "strategy": {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy.signal_strategy",
            "kwargs": {
                "signal": (model, dataset),
                "topk": 50,
                "n_drop": 5,
            },
        },
        "backtest": {
            "start_time": "2017-01-01",
            "end_time": "2020-08-01",
            "account": 100000000,
            "benchmark": CSI300_BENCH,
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
    # backtest
    par = PortAnaRecord(recorder, port_analysis_config, risk_analysis_freq="day")
    par.generate()
    analysis_df = par.load("port_analysis_1day.pkl")
    print(analysis_df)
    return analysis_df


class TestAllFlow(TestAutoData):
    REPORT_NORMAL = None
    POSITIONS = None
    RID = None
    URI_PATH = "file:" + str(Path(__file__).parent.joinpath("test_all_flow_mlruns").resolve())

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.URI_PATH.lstrip("file:"))

    @pytest.mark.slow
    def test_0_train(self):
        TestAllFlow.PRED_SCORE, ic_ric, TestAllFlow.RID, uri_path = train(self.URI_PATH)
        self.assertGreaterEqual(ic_ric["ic"].all(), 0, "train failed")
        self.assertGreaterEqual(ic_ric["ric"].all(), 0, "train failed")

    @pytest.mark.slow
    def test_1_backtest(self):
        analyze_df = backtest_analysis(TestAllFlow.PRED_SCORE, TestAllFlow.RID, self.URI_PATH)
        self.assertGreaterEqual(
            analyze_df.loc(axis=0)["excess_return_with_cost", "annualized_return"].values[0],
            0.05,
            "backtest failed",
        )
        self.assertTrue(not analyze_df.isna().any().any(), "backtest failed")

    @pytest.mark.slow
    def test_2_expmanager(self):
        pass_default, pass_current, uri_path = fake_experiment()
        self.assertTrue(pass_default, msg="default uri is incorrect")
        self.assertTrue(pass_current, msg="current uri is incorrect")
        shutil.rmtree(str(Path(uri_path.strip("file:")).resolve()))


def suite():
    _suite = unittest.TestSuite()
    _suite.addTest(TestAllFlow("test_0_train"))
    _suite.addTest(TestAllFlow("test_1_backtest"))
    _suite.addTest(TestAllFlow("test_2_expmanager"))
    return _suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
