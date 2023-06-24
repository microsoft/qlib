import logging
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

from qlib.utils import class_casting

from ..data.dataset import DatasetH
from ..data.dataset.handler import DataHandlerLP
from ..log import get_module_logger
from ..contrib.eva.alpha import calc_ic, calc_long_short_return, calc_long_short_prec

logger = get_module_logger("analysis", logging.INFO)


class AnalyzerTemp:
    def __init__(self, workspace=None, **kwargs):
        self.workspace = Path(workspace) if workspace else "./"

    def analyse(self, **kwargs):
        """
        Analyse data index, distribution .etc

        Parameters
        ----------


        Return
        ------
        The handled data.
        """
        raise NotImplementedError(f"Please implement the `analysis` method.")


class HFAnalyzer(AnalyzerTemp):
    """
    This is the Signal Analysis class that generates the analysis results such as IC and IR.

    default output image filename is "HFAnalyzerTable.jpeg"
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def analyse(self, pred=None, label=None):
        long_pre, short_pre = calc_long_short_prec(pred.iloc[:, 0], label.iloc[:, 0], is_alpha=True)
        ic, ric = calc_ic(pred.iloc[:, 0], label.iloc[:, 0])
        metrics = {
            "IC": ic.mean(),
            "ICIR": ic.mean() / ic.std(),
            "Rank IC": ric.mean(),
            "Rank ICIR": ric.mean() / ric.std(),
            "Long precision": long_pre.mean(),
            "Short precision": short_pre.mean(),
        }

        long_short_r, long_avg_r = calc_long_short_return(pred.iloc[:, 0], label.iloc[:, 0])
        metrics.update(
            {
                "Long-Short Average Return": long_short_r.mean(),
                "Long-Short Average Sharpe": long_short_r.mean() / long_short_r.std(),
            }
        )

        table = [[k, v] for (k, v) in metrics.items()]
        plt.table(cellText=table, loc="center")
        plt.axis("off")
        plt.savefig(self.workspace.joinpath("HFAnalyzerTable.jpeg"))
        plt.clf()

        plt.scatter(np.arange(0, len(pred)), pred.iloc[:, 0])
        plt.scatter(np.arange(0, len(label)), label.iloc[:, 0])
        plt.title("HFAnalyzer")
        plt.savefig(self.workspace.joinpath("HFAnalyzer.jpeg"))
        return "HFAnalyzer.jpeg"


class SignalAnalyzer(AnalyzerTemp):
    """
    This is the Signal Analysis class that generates the analysis results such as IC and IR.

    default output image filename is "signalAnalysis.jpeg"
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def analyse(self, dataset=None, **kwargs):

        with class_casting(dataset, DatasetH):
            params = dict(segments="test", col_set="label", data_key=DataHandlerLP.DK_R)
            try:
                # Assume the backend handler is DataHandlerLP
                raw_label = dataset.prepare(**params)
            except TypeError:
                # The argument number is not right
                del params["data_key"]
                # The backend handler should be DataHandler
                raw_label = dataset.prepare(**params)
            except AttributeError as e:
                # The data handler is initialized with `drop_raw=True`...
                # So raw_label is not available
                logger.warning(f"Exception: {e}")
                raw_label = None
        plt.hist(raw_label)
        plt.title("SignalAnalyzer")
        plt.savefig(self.workspace.joinpath("signalAnalysis.jpeg"))

        return "signalAnalysis.jpeg"
