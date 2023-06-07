import logging
import matplotlib.pyplot as plt

from qlib.utils import class_casting

from ..data.dataset import DatasetH
from ..data.dataset.handler import DataHandlerLP
from ..log import get_module_logger

logger = get_module_logger("analysis", logging.INFO)


class AnalysisTemp:

    def __init__(self, **kwargs):
        self.workspace = './'
        pass

    def analysis(self, **kwargs):
        raise NotImplementedError(f"Please implement the `analysis` method.")


class HFAnalysis(AnalysisTemp):
    """
    This is the Signal Analysis class that generates the analysis results such as IC and IR.
    This class inherits the ``AnalysisTemp`` class.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def analysis(self, pred=None, label=None):
        plt.plot(pred, label)
        plt.show()


class SignalAnalysis(AnalysisTemp):
    """
    This is the Signal Analysis class that generates the analysis results such as IC and IR.
    This class inherits the ``AnalysisTemp`` class.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def analysis(self, dataset=None, **kwargs):

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
        plt.show()
        plt.savefig('signalAnalysis.jpeg')
        return raw_label
