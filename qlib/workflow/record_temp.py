#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import re
import pandas as pd
from pathlib import Path
from pprint import pprint
from ..contrib.evaluate import (
    backtest as normal_backtest,
    risk_analysis,
)
from ..data.dataset import DatasetH
from ..data.dataset.handler import DataHandlerLP
from ..utils import init_instance_by_config, get_module_by_module_path
from ..log import get_module_logger
from ..utils import flatten_dict
from ..contrib.eva.alpha import calc_ic

logger = get_module_logger("workflow", "INFO")


class RecordTemp:
    """
    This is the Records Template class that enables user to generate experiment results such as IC and
    backtest in a certain format.
    """
    artifact_path = None

    @classmethod
    def get_path(cls, path=None):
        names = []
        if cls.artifact_path is not None:
            names.append(cls.artifact_path)

        if path is not None:
            names.append(path)

        return "/".join(names)

    def __init__(self, recorder):
        self.recorder = recorder

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

    def load(self, name):
        """
        Load the stored records.

        Parameters
        ----------
        name : str
            the name for the file to be load.

        Return
        ------
        The stored records.
        """
        # try to load the saved object
        obj = self.recorder.load_object(name)
        return obj

    def list(self):
        """
        List the stored records.

        Return
        ------
        A list of all the stored records.
        """
        return []

    def check(self, parent=False):
        """
        Check if the records is properly generated and saved.

        Raise
        ------
        FileExistsError: whether the records are stored properly.
        """
        artifacts = set(self.recorder.list_artifacts())
        if parent:
            # Downcasting have to be done here instead of using `super`
            flist = self.__class__.__base__.list(self)  # pylint: disable=E1101
        else:
            flist = self.list()
        for item in flist:
            if item not in artifacts:
                raise FileExistsError(item)


class SignalRecord(RecordTemp):
    """
    This is the Signal Record class that generates the signal prediction.
    """

    def __init__(self, model=None, dataset=None, recorder=None, **kwargs):
        super().__init__(recorder=recorder)
        self.model = model
        self.dataset = dataset

    def generate(self, **kwargs):
        # generate prediciton
        pred = self.model.predict(self.dataset)
        if isinstance(pred, pd.Series):
            pred = pred.to_frame("score")
        self.recorder.save_objects(**{"pred.pkl": pred})

        logger.info(
            f"Signal record 'pred.pkl' has been saved as the artifact of the Experiment {self.recorder.experiment_id}"
        )
        # print out results
        pprint(f"The following are prediction results of the {type(self.model).__name__} model.")
        pprint(pred.head(5))

        # save according label
        if isinstance(self.dataset, DatasetH):
            params = dict(self=self.dataset, segments="test", col_set="label", data_key=DataHandlerLP.DK_R)
            try:
                # Assume the backend handler is DataHandlerLP
                raw_label = DatasetH.prepare(**params)
            except TypeError:
                # The argument number is not right
                del params["data_key"]
                # The backend handler should be DataHandler
                raw_label = DatasetH.prepare(**params)
            self.recorder.save_objects(**{"label.pkl": raw_label})

    def list(self):
        return ["pred.pkl", "label.pkl"]

    def load(self, name="pred.pkl"):
        return super().load(name)


class SigAnaRecord(SignalRecord):

    artifact_path = "sig_analysis"

    def __init__(self, recorder, **kwargs):
        super().__init__(recorder=recorder, **kwargs)
        # The name must be unique. Otherwise it will be overridden

    def generate(self):
        self.check(parent=True)

        pred = self.load("pred.pkl")
        label = self.load("label.pkl")
        ic, ric = calc_ic(pred.iloc[:, 0], label.iloc[:, 0])
        metrics = {
            "IC": ic.mean(),
            "ICIR": ic.mean() / ic.std(),
            "Rank IC": ric.mean(),
            "Rank ICIR": ric.mean() / ric.std(),
        }
        self.recorder.log_metrics(**metrics)
        self.recorder.save_objects(**{"ic.pkl": ic, "ric.pkl": ric}, artifact_path=self.get_path())
        pprint(metrics)

    def list(self):
        return [self.get_path("ic.pkl"), self.get_path("ric.pkl")]


class PortAnaRecord(SignalRecord):
    """
    This is the Portfolio Analysis Record class that generates the results such as those of backtest.
    """

    artifact_path = "portfolio_analysis"

    def __init__(self, recorder, config, **kwargs):
        """
        config["strategy"] : dict
            define the strategy class as well as the kwargs.
        config["backtest"] : dict
            define the backtest kwargs.
        """
        super().__init__(recorder=recorder)

        self.strategy_config = config["strategy"]
        self.backtest_config = config["backtest"]
        self.strategy = init_instance_by_config(self.strategy_config)

    def generate(self, **kwargs):
        # check previously stored prediction results
        self.check(parent=True)  # "Make sure the parent process is completed and store the data properly."

        # custom strategy and get backtest
        pred_score = super().load()
        report_normal, positions_normal = normal_backtest(pred_score, strategy=self.strategy, **self.backtest_config)
        self.recorder.save_objects(**{"report_normal.pkl": report_normal}, artifact_path=PortAnaRecord.get_path())
        self.recorder.save_objects(**{"positions_normal.pkl": positions_normal}, artifact_path=PortAnaRecord.get_path())

        # analysis
        analysis = dict()
        analysis["excess_return_without_cost"] = risk_analysis(report_normal["return"] - report_normal["bench"])
        analysis["excess_return_with_cost"] = risk_analysis(
            report_normal["return"] - report_normal["bench"] - report_normal["cost"]
        )
        # save portfolio analysis results
        analysis_df = pd.concat(analysis)  # type: pd.DataFrame
        # log metrics
        self.recorder.log_metrics(**flatten_dict(analysis_df["risk"].unstack().T.to_dict()))
        # save results
        self.recorder.save_objects(**{"port_analysis.pkl": analysis_df}, artifact_path=PortAnaRecord.get_path())
        logger.info(
            f"Portfolio analysis record 'port_analysis.pkl' has been saved as the artifact of the Experiment {self.recorder.experiment_id}"
        )
        # print out results
        pprint("The following are analysis results of the excess return without cost.")
        pprint(analysis["excess_return_without_cost"])
        pprint("The following are analysis results of the excess return with cost.")
        pprint(analysis["excess_return_with_cost"])

    def list(self):
        return [
            PortAnaRecord.get_path("report_normal.pkl"),
            PortAnaRecord.get_path("positions_normal.pkl"),
            PortAnaRecord.get_path("port_analysis.pkl"),
        ]
