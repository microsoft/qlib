#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import re
import logging
import warnings
import pandas as pd
from pathlib import Path
from pprint import pprint
from typing import Union, List
from ..contrib.evaluate import indicator_analysis, risk_analysis, indicator_analysis

from ..data.dataset import DatasetH
from ..data.dataset.handler import DataHandlerLP
from ..backtest import backtest as normal_backtest
from ..utils import init_instance_by_config, get_module_by_module_path
from ..log import get_module_logger
from ..utils import flatten_dict
from ..utils.time import Freq
from ..strategy.base import BaseStrategy
from ..contrib.eva.alpha import calc_ic, calc_long_short_return, calc_long_short_prec


logger = get_module_logger("workflow", logging.INFO)


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
        self._recorder = recorder

    @property
    def recorder(self):
        if self._recorder is None:
            raise ValueError("This RecordTemp did not set recorder yet.")
        return self._recorder

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
        Load the stored records. Due to the fact that some problems occured when we tried to balancing a clean API
        with the Python's inheritance. This method has to be used in a rather ugly way, and we will try to fix them
        in the future::

            sar = SigAnaRecord(recorder)
            ic = sar.load(sar.get_path("ic.pkl"))

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
        List the supported artifacts.

        Return
        ------
        A list of all the supported artifacts.
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
    This is the Signal Record class that generates the signal prediction. This class inherits the ``RecordTemp`` class.
    """

    def __init__(self, model=None, dataset=None, recorder=None):
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

        if isinstance(self.dataset, DatasetH):
            # NOTE:
            # Python doesn't provide the downcasting mechanism.
            # We use the trick here to downcast the class
            orig_cls = self.dataset.__class__
            self.dataset.__class__ = DatasetH

            params = dict(segments="test", col_set="label", data_key=DataHandlerLP.DK_R)
            try:
                # Assume the backend handler is DataHandlerLP
                raw_label = self.dataset.prepare(**params)
            except TypeError:
                # The argument number is not right
                del params["data_key"]
                # The backend handler should be DataHandler
                raw_label = self.dataset.prepare(**params)
            except AttributeError:
                # The data handler is initialize with `drop_raw=True`...
                # So raw_label is not available
                raw_label = None

            self.recorder.save_objects(**{"label.pkl": raw_label})
            self.dataset.__class__ = orig_cls

    def list(self):
        return ["pred.pkl", "label.pkl"]

    def load(self, name="pred.pkl"):
        return super().load(name)


class HFSignalRecord(SignalRecord):
    """
    This is the Signal Analysis Record class that generates the analysis results such as IC and IR. This class inherits the ``RecordTemp`` class.
    """

    artifact_path = "hg_sig_analysis"

    def __init__(self, recorder, **kwargs):
        super().__init__(recorder=recorder)

    def generate(self):
        pred = self.load("pred.pkl")
        raw_label = self.load("label.pkl")
        long_pre, short_pre = calc_long_short_prec(pred.iloc[:, 0], raw_label.iloc[:, 0], is_alpha=True)
        ic, ric = calc_ic(pred.iloc[:, 0], raw_label.iloc[:, 0])
        metrics = {
            "IC": ic.mean(),
            "ICIR": ic.mean() / ic.std(),
            "Rank IC": ric.mean(),
            "Rank ICIR": ric.mean() / ric.std(),
            "Long precision": long_pre.mean(),
            "Short precision": short_pre.mean(),
        }
        objects = {"ic.pkl": ic, "ric.pkl": ric}
        objects.update({"long_pre.pkl": long_pre, "short_pre.pkl": short_pre})
        long_short_r, long_avg_r = calc_long_short_return(pred.iloc[:, 0], raw_label.iloc[:, 0])
        metrics.update(
            {
                "Long-Short Average Return": long_short_r.mean(),
                "Long-Short Average Sharpe": long_short_r.mean() / long_short_r.std(),
            }
        )
        objects.update(
            {
                "long_short_r.pkl": long_short_r,
                "long_avg_r.pkl": long_avg_r,
            }
        )
        self.recorder.log_metrics(**metrics)
        self.recorder.save_objects(**objects, artifact_path=self.get_path())
        pprint(metrics)

    def list(self):
        paths = [
            self.get_path("ic.pkl"),
            self.get_path("ric.pkl"),
            self.get_path("long_pre.pkl"),
            self.get_path("short_pre.pkl"),
            self.get_path("long_short_r.pkl"),
            self.get_path("long_avg_r.pkl"),
        ]
        return paths


class SigAnaRecord(SignalRecord):
    """
    This is the Signal Analysis Record class that generates the analysis results such as IC and IR. This class inherits the ``RecordTemp`` class.
    """

    artifact_path = "sig_analysis"

    def __init__(self, recorder, ana_long_short=False, ann_scaler=252, **kwargs):
        super().__init__(recorder=recorder, **kwargs)
        self.ana_long_short = ana_long_short
        self.ann_scaler = ann_scaler

    def generate(self, **kwargs):
        try:
            self.check(parent=True)
        except FileExistsError:
            super().generate()

        pred = self.load("pred.pkl")
        label = self.load("label.pkl")
        if label is None or not isinstance(label, pd.DataFrame) or label.empty:
            logger.warn(f"Empty label.")
            return
        ic, ric = calc_ic(pred.iloc[:, 0], label.iloc[:, 0])
        metrics = {
            "IC": ic.mean(),
            "ICIR": ic.mean() / ic.std(),
            "Rank IC": ric.mean(),
            "Rank ICIR": ric.mean() / ric.std(),
        }
        objects = {"ic.pkl": ic, "ric.pkl": ric}
        if self.ana_long_short:
            long_short_r, long_avg_r = calc_long_short_return(pred.iloc[:, 0], label.iloc[:, 0])
            metrics.update(
                {
                    "Long-Short Ann Return": long_short_r.mean() * self.ann_scaler,
                    "Long-Short Ann Sharpe": long_short_r.mean() / long_short_r.std() * self.ann_scaler ** 0.5,
                    "Long-Avg Ann Return": long_avg_r.mean() * self.ann_scaler,
                    "Long-Avg Ann Sharpe": long_avg_r.mean() / long_avg_r.std() * self.ann_scaler ** 0.5,
                }
            )
            objects.update(
                {
                    "long_short_r.pkl": long_short_r,
                    "long_avg_r.pkl": long_avg_r,
                }
            )
        self.recorder.log_metrics(**metrics)
        self.recorder.save_objects(**objects, artifact_path=self.get_path())
        pprint(metrics)

    def list(self):
        paths = [self.get_path("ic.pkl"), self.get_path("ric.pkl")]
        if self.ana_long_short:
            paths.extend([self.get_path("long_short_r.pkl"), self.get_path("long_avg_r.pkl")])
        return paths


class PortAnaRecord(RecordTemp):
    """
    This is the Portfolio Analysis Record class that generates the analysis results such as those of backtest. This class inherits the ``RecordTemp`` class.

    The following files will be stored in recorder
    - report_normal.pkl & positions_normal.pkl:
        - The return report and detailed positions of the backtest, returned by `qlib/contrib/evaluate.py:backtest`
    - port_analysis.pkl : The risk analysis of your portfolio, returned by `qlib/contrib/evaluate.py:risk_analysis`
    """

    artifact_path = "portfolio_analysis"

    def __init__(
        self,
        recorder,
        config,
        risk_analysis_freq: Union[List, str] = None,
        indicator_analysis_freq: Union[List, str] = None,
        indicator_analysis_method=None,
        **kwargs,
    ):
        """
        config["strategy"] : dict
            define the strategy class as well as the kwargs.
        config["executor"] : dict
            define the executor class as well as the kwargs.
        config["backtest"] : dict
            define the backtest kwargs.
        risk_analysis_freq : str|List[str]
            risk analysis freq of report
        indicator_analysis_freq : str|List[str]
            indicator analysis freq of report
        indicator_analysis_method : str, optional, default by None
            the candidated values include 'mean', 'amount_weighted', 'value_weighted'
        """
        super().__init__(recorder=recorder, **kwargs)

        self.strategy_config = config["strategy"]
        _default_executor_config = {
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {
                "time_per_step": "day",
                "generate_report": True,
            },
        }
        self.executor_config = config.get("executor", _default_executor_config)
        self.backtest_config = config["backtest"]

        self.all_freq = self._get_report_freq(self.executor_config)
        if risk_analysis_freq is None:
            risk_analysis_freq = [self.all_freq[0]]
        if indicator_analysis_freq is None:
            indicator_analysis_freq = [self.all_freq[0]]

        if isinstance(risk_analysis_freq, str):
            risk_analysis_freq = [risk_analysis_freq]
        if isinstance(indicator_analysis_freq, str):
            indicator_analysis_freq = [indicator_analysis_freq]

        self.risk_analysis_freq = [
            "{0}{1}".format(*Freq.parse(_analysis_freq)) for _analysis_freq in risk_analysis_freq
        ]
        self.indicator_analysis_freq = [
            "{0}{1}".format(*Freq.parse(_analysis_freq)) for _analysis_freq in indicator_analysis_freq
        ]
        self.indicator_analysis_method = indicator_analysis_method

    def _get_report_freq(self, executor_config):
        ret_freq = []
        if executor_config["kwargs"].get("generate_report", False):
            _count, _freq = Freq.parse(executor_config["kwargs"]["time_per_step"])
            ret_freq.append(f"{_count}{_freq}")
        if "sub_env" in executor_config["kwargs"]:
            ret_freq.extend(self._get_report_freq(executor_config["kwargs"]["sub_env"]))
        return ret_freq

    def generate(self, **kwargs):
        # custom strategy and get backtest
        report_dict, indicator_dict = normal_backtest(
            executor=self.executor_config, strategy=self.strategy_config, **self.backtest_config
        )
        for _freq, (report_normal, positions_normal) in report_dict.items():
            self.recorder.save_objects(
                **{f"report_normal_{_freq}.pkl": report_normal}, artifact_path=PortAnaRecord.get_path()
            )
            self.recorder.save_objects(
                **{f"positions_normal_{_freq}.pkl": positions_normal}, artifact_path=PortAnaRecord.get_path()
            )

        for _freq, indicators_normal in indicator_dict.items():
            self.recorder.save_objects(
                **{f"indicators_normal_{_freq}.pkl": indicators_normal}, artifact_path=PortAnaRecord.get_path()
            )

        for _analysis_freq in self.risk_analysis_freq:
            if _analysis_freq not in report_dict:
                warnings.warn(
                    f"the freq {_analysis_freq} report is not found, please set the corresponding env with `generate_report=True`"
                )
            else:
                report_normal, _ = report_dict.get(_analysis_freq)
                analysis = dict()
                analysis["excess_return_without_cost"] = risk_analysis(
                    report_normal["return"] - report_normal["bench"], freq=_analysis_freq
                )
                analysis["excess_return_with_cost"] = risk_analysis(
                    report_normal["return"] - report_normal["bench"] - report_normal["cost"], freq=_analysis_freq
                )

                analysis_df = pd.concat(analysis)  # type: pd.DataFrame
                # log metrics
                analysis_dict = flatten_dict(analysis_df["risk"].unstack().T.to_dict())
                self.recorder.log_metrics(**{f"{_analysis_freq}.{k}": v for k, v in analysis_dict.items()})
                # save results
                self.recorder.save_objects(
                    **{f"port_analysis_{_analysis_freq}.pkl": analysis_df}, artifact_path=PortAnaRecord.get_path()
                )
                logger.info(
                    f"Portfolio analysis record 'port_analysis_{_analysis_freq}.pkl' has been saved as the artifact of the Experiment {self.recorder.experiment_id}"
                )
                # print out results
                pprint(f"The following are analysis results of benchmark return({_analysis_freq}).")
                pprint(risk_analysis(report_normal["bench"], freq=_analysis_freq))
                pprint(f"The following are analysis results of the excess return without cost({_analysis_freq}).")
                pprint(analysis["excess_return_without_cost"])
                pprint(f"The following are analysis results of the excess return with cost({_analysis_freq}).")
                pprint(analysis["excess_return_with_cost"])

        for _analysis_freq in self.indicator_analysis_freq:
            if _analysis_freq not in indicator_dict:
                warnings.warn(f"the freq {_analysis_freq} indicator is not found")
            else:
                indicators_normal = indicator_dict.get(_analysis_freq)
                if self.indicator_analysis_method is None:
                    analysis_df = indicator_analysis(indicators_normal)
                else:
                    analysis_df = indicator_analysis(indicators_normal, method=self.indicator_analysis_method)
                # log metrics
                analysis_dict = analysis_df["value"].to_dict()
                self.recorder.log_metrics(**{f"{_analysis_freq}.{k}": v for k, v in analysis_dict.items()})
                # save results
                self.recorder.save_objects(
                    **{f"indicator_analysis_{_analysis_freq}.pkl": analysis_df}, artifact_path=PortAnaRecord.get_path()
                )
                logger.info(
                    f"Indicator analysis record 'indicator_analysis_{_analysis_freq}.pkl' has been saved as the artifact of the Experiment {self.recorder.experiment_id}"
                )
                pprint(f"The following are analysis results of indicators({_analysis_freq}).")
                pprint(analysis_df)

    def list(self):
        list_path = []
        for _freq in self.all_freq:
            list_path.extend(
                [
                    PortAnaRecord.get_path(f"report_normal_{_freq}.pkl"),
                    PortAnaRecord.get_path(f"positions_normal_{_freq}.pkl"),
                ]
            )
        for _analysis_freq in self.risk_analysis_freq:
            if _analysis_freq in self.all_freq:
                list_path.append(PortAnaRecord.get_path(f"port_analysis_{_analysis_freq}.pkl"))
            else:
                warnings.warn(f"risk_analysis freq {_analysis_freq} is not found")

        for _analysis_freq in self.indicator_analysis_freq:
            if _analysis_freq in self.all_freq:
                list_path.append(PortAnaRecord.get_path(f"indicator_analysis_{_analysis_freq}.pkl"))
            else:
                warnings.warn(f"indicator_analysis freq {_analysis_freq} is not found")

        return list_path
