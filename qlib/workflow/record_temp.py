#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import pandas as pd
from pathlib import Path
from pprint import pprint
from ..contrib.evaluate import (
    backtest as normal_backtest,
    risk_analysis,
)
from ..utils import init_instance_by_config, get_module_by_module_path
from ..log import get_module_logger

logger = get_module_logger("workflow", "INFO")


class RecordTemp:
    """
    This is the Records Template class that enables user to generate experiment results such as IC and
    backtest in a certain format.
    """

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
        """
        raise NotImplementedError(f"Please implement the `generate` method.")

    def load(self, **kwargs):
        """
        Load the stored records.

        Parameters
        ----------
        kwargs

        Return
        ------
        The stored records.
        """
        raise NotImplementedError(f"Please implement the `load` method.")

    def check(self, **kwargs):
        """
        Check if the records is properly generated and saved.

        Parameters
        ----------
        kwargs

        Return
        ------
        Boolean: whether the records are stored properly.
        """
        raise NotImplementedError(f"Please implement the `check` method.")


# TODO: this can only be run under R's running experiment.
class SignalRecord(RecordTemp):
    """
    This is the Signal Record class that generates the signal prediction.
    """

    def __init__(self, model, dataset, recorder, **kwargs):
        super(SignalRecord, self).__init__()
        self.model = model
        self.dataset = dataset
        self.recorder = recorder

    def generate(self, **kwargs):
        # generate prediciton
        pred = self.model.predict(self.dataset)
        self.recorder.save_objects(**{"pred.pkl": pred})
        logger.info(
            f"Signal record 'pred.pkl' has been saved as the artifact of the Experiment {self.recorder.experiment_id}"
        )
        # print out results
        pprint(f"The following are prediction results of the {type(self.model).__name__} model.")
        pprint(pred.head(5))

    def load(self):
        # try to load the saved object
        pred = self.recorder.load_object("pred.pkl")
        return pred

    def check(self, **kwargs):
        artifacts = self.recorder.list_artifacts()
        for artifact in artifacts:
            if "pred.pkl" in artifact.path:
                return True
        return False


# TODO
class SigAnaRecord(SignalRecord):
    def __init__(self, recorder, config, **kwargs):
        pass

    def generate(self):
        pass

    def load(self):
        pass

    def check(self):
        pass


class PortAnaRecord(SignalRecord):
    """
    This is the Portfolio Analysis Record class that generates the results such as those of backtest.
    """

    def __init__(self, recorder, config, **kwargs):
        self.recorder = recorder
        self.strategy_config = config["strategy"]
        self.backtest_config = config["backtest"]
        self.strategy = init_instance_by_config(self.strategy_config)
        self.artifact_path = "portfolio_analysis"

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
        report_normal, positions_normal = normal_backtest(pred_score, strategy=self.strategy, **self.backtest_config)
        self.recorder.save_objects(**{"report_normal.pkl": report_normal}, artifact_path=self.artifact_path)
        self.recorder.save_objects(**{"positions_normal.pkl": positions_normal}, artifact_path=self.artifact_path)

        # analysis
        analysis = dict()
        analysis["excess_return_without_cost"] = risk_analysis(report_normal["return"] - report_normal["bench"])
        analysis["excess_return_with_cost"] = risk_analysis(
            report_normal["return"] - report_normal["bench"] - report_normal["cost"]
        )
        # log metrics
        self.recorder.log_metrics(
            excess_return_without_cost_mean=analysis["excess_return_without_cost"]["risk"]["mean"]
        )
        self.recorder.log_metrics(excess_return_without_cost_std=analysis["excess_return_without_cost"]["risk"]["std"])
        self.recorder.log_metrics(
            excess_return_without_cost_annualized_return=analysis["excess_return_without_cost"]["risk"][
                "annualized_return"
            ]
        )
        self.recorder.log_metrics(
            excess_return_without_cost_information_ratio=analysis["excess_return_without_cost"]["risk"][
                "information_ratio"
            ]
        )
        self.recorder.log_metrics(
            excess_return_without_cost_max_drawdown=analysis["excess_return_without_cost"]["risk"]["max_drawdown"]
        )
        self.recorder.log_metrics(excess_return_with_cost_mean=analysis["excess_return_with_cost"]["risk"]["mean"])
        self.recorder.log_metrics(excess_return_with_cost_std=analysis["excess_return_with_cost"]["risk"]["std"])
        self.recorder.log_metrics(
            excess_return_with_cost_annualized_return=analysis["excess_return_with_cost"]["risk"]["annualized_return"]
        )
        self.recorder.log_metrics(
            excess_return_with_cost_information_ratio=analysis["excess_return_with_cost"]["risk"]["information_ratio"]
        )
        self.recorder.log_metrics(
            excess_return_with_cost_max_drawdown=analysis["excess_return_with_cost"]["risk"]["max_drawdown"]
        )
        # save portfolio analysis results
        analysis_df = pd.concat(analysis)  # type: pd.DataFrame
        self.recorder.save_objects(**{"port_analysis.pkl": analysis_df}, artifact_path=self.artifact_path)
        logger.info(
            f"Portfolio analysis record 'port_analysis.pkl' has been saved as the artifact of the Experiment {self.recorder.experiment_id}"
        )
        # print out results
        pprint("The following are analysis results of the excess return without cost.")
        pprint(analysis["excess_return_without_cost"])
        pprint("The following are analysis results of the excess return with cost.")
        pprint(analysis["excess_return_with_cost"])

    def load(self):
        # try to load the saved object
        pred = self.recorder.load_object(self.artifact_path / "port_analysis.pkl")
        return pred

    def check(self):
        artifacts = self.recorder.list_artifacts(self.artifact_path)
        for artifact in artifacts:
            if "port_analysis.pkl" in artifact.path:
                return True
        return False
