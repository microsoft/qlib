import logging
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

from ..log import get_module_logger
from ..contrib.eva.alpha import calc_ic, calc_long_short_return, calc_long_short_prec

logger = get_module_logger("analysis", logging.INFO)


class AnalyzerTemp:
    def __init__(self, recorder, output_dir=None, **kwargs):
        self.recorder = recorder
        self.output_dir = Path(output_dir) if output_dir else "./"

    def load(self, name: str):
        """
        It behaves the same as self.recorder.load_object.
        But it is an easier interface because users don't have to care about `get_path` and `artifact_path`

        Parameters
        ----------
        name : str
            the name for the file to be load.

        Return
        ------
        The stored records.
        """
        return self.recorder.load_object(name)

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

    def analyse(self):
        pred = self.load("pred.pkl")
        label = self.load("label.pkl")

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
        plt.savefig(self.output_dir.joinpath("HFAnalyzerTable.jpeg"))
        plt.clf()

        plt.scatter(np.arange(0, len(pred)), pred.iloc[:, 0])
        plt.scatter(np.arange(0, len(label)), label.iloc[:, 0])
        plt.title("HFAnalyzer")
        plt.savefig(self.output_dir.joinpath("HFAnalyzer.jpeg"))
        return "HFAnalyzer.jpeg"


class SignalAnalyzer(AnalyzerTemp):
    """
    This is the Signal Analysis class that generates the analysis results such as IC and IR.

    default output image filename is "signalAnalysis.jpeg"
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def analyse(self, dataset=None, **kwargs):
        label = self.load("label.pkl")

        plt.hist(label)
        plt.title("SignalAnalyzer")
        plt.savefig(self.output_dir.joinpath("signalAnalysis.jpeg"))

        return "signalAnalysis.jpeg"
