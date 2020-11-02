#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import pandas as pd
from pathlib import Path
from ..contrib.evaluate import (
    backtest as normal_backtest,
    risk_analysis,
)
from ..utils import init_instance_by_config, get_module_by_module_path


class RecordTemp:
    def __init__(self, *args, **kwargs):
        pass

    def generate(self, **kwargs):
        """
        Generate certain records such as IC, backtest etc., and save them.

        Parameters
        ----------
        kwargs

        Return
        ------
        The generated records.
        """
        raise NotImplementedError(f"Please implement the `generate` method.")

    def check(self, **kwargs):
        """
        Check if the records is properly generated and saved.

        Parameters
        ----------
        kwargs
        """
        raise NotImplementedError(f"Please implement the `check` method.")


# TODO: this can only be run under R's running experiment.
class SignalRecord(RecordTemp):
    def __init__(self, model, dataset, recorder, **kwargs):
        super(SignalRecord, self).__init__()
        self.model = model
        self.dataset = dataset
        self.recorder = recorder

    def generate(self, **kwargs):
        # generate prediciton
        pred = self.model.predict(self.dataset)
        self.recorder.save_object(pred, "pred.pkl")

    def load(self):
        # try to load the saved object
        try:
            pred = self.recorder.load_object("pred.pkl")
            return pred
        except:
            raise Exception("Something went wrong when loading the saved object.")

    def check(self, **kwargs):
        return self.recorder.check("pred.pkl")


# TODO
class SigAnaRecord(SignalRecord):
    def __init__(self, recorder, **kwargs):
        pass

    def generate(self):
        pass

    def load(self):
        pass

    def check(self):
        pass


class PortAnaRecord(SignalRecord):
    def __init__(self, recorder, STRATEGY_CONFIG, BACKTEST_CONFIG, **kwargs):
        self.recorder = recorder
        self.STRATEGY_CONFIG = STRATEGY_CONFIG
        self.BACKTEST_CONFIG = BACKTEST_CONFIG
        module = get_module_by_module_path("qlib.contrib.strategy")
        self.strategy = init_instance_by_config(STRATEGY_CONFIG, module)
        self.artifact_path = Path("portfolio_analysis").resolve()

    def generate(self, **kwargs):
        """
        STRATEGY_CONFIG : dict
            define the strategy class as well as the kwargs.
        BACKTEST_CONFIG : dict
            define the backtest kwargs.
        """
        # check previously stored prediction results
        assert super().check(), "Make sure the parent process is completed and store the data properly."
        # custom strategy and get backtest
        pred_score = super().load()
        report_normal, positions_normal = normal_backtest(pred_score, strategy=self.strategy, **self.BACKTEST_CONFIG)
        self.recorder.save_object(report_normal, "report_normal.pkl", self.artifact_path)
        self.recorder.save_object(positions_normal, "positions_normal.pkl", self.artifact_path)

        # analysis
        analysis = dict()
        analysis["excess_return_without_cost"] = risk_analysis(report_normal["return"] - report_normal["bench"])
        analysis["excess_return_with_cost"] = risk_analysis(
            report_normal["return"] - report_normal["bench"] - report_normal["cost"]
        )
        analysis_df = pd.concat(analysis)  # type: pd.DataFrame
        self.recorder.save_object(pred, "port_analysis.pkl", self.artifact_path)

    def load(self):
        # try to load the saved object
        try:
            pred = self.recorder.load_object(self.artifact_path / "port_analysis.pkl")
            return pred
        except:
            raise Exception("Something went wrong when loading the saved object.")

    def check(self):
        return self.recorder.check("port_analysis.pkl", self.artifact_path)
