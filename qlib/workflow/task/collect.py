from abc import abstractmethod
from typing import Callable, Union
from qlib.workflow.task.utils import list_recorders
from qlib.utils.serial import Serializable


class Collector:
    """The collector to collect different results"""

    def collect(self, *args, **kwargs):
        """Collect the results and return a dict like {key: things}

        Returns:
            dict: the dict after collected.

            For example:

            {"prediction": pd.Series}

            {"IC": {"Xgboost": pd.Series, "LSTM": pd.Series}}

            ......
        """
        raise NotImplementedError(f"Please implement the `collect` method.")


class RecorderCollector(Collector):
    ART_KEY_RAW = "__raw"

    def __init__(
        self,
        exp_name,
        rec_key_func=None,
        rec_filter_func=None,
        artifacts_path={"pred": "pred.pkl", "IC": "sig_analysis/ic.pkl"},
        artifacts_key=None,
    ):
        """init RecorderCollector

        Args:
            exp_name (str): the name of Experiment
            rec_key_func (Callable): a function to get the key of a recorder. If None, use recorder id.
            rec_filter_func (Callable, optional): filter the recorder by return True or False. Defaults to None.
            artifacts_path (dict, optional): The artifacts name and its path in Recorder. Defaults to {"pred": "pred.pkl", "IC": "sig_analysis/ic.pkl"}.
            artifacts_key (str or List, optional): the artifacts key you want to get. If None, get all artifacts.
        """
        self.exp_name = exp_name
        self.artifacts_path = artifacts_path
        if rec_key_func is None:
            rec_key_func = lambda rec: rec.info["id"]
        if artifacts_key is None:
            artifacts_key = self.artifacts_path.keys()
        self._rec_key_func = rec_key_func
        self.artifacts_key = artifacts_key
        self._rec_filter_func = rec_filter_func

    def collect(self, artifacts_key=None, rec_filter_func=None):
        """Collect different artifacts based on recorder after filtering.

        Args:
            artifacts_key (str or List, optional): the artifacts key you want to get. If None, use default.
            rec_filter_func (Callable, optional): filter the recorder by return True or False. If None, use default.

        Returns:
            dict: the dict after collected like {artifact: {rec_key: object}}
        """
        if artifacts_key is None:
            artifacts_key = self.artifacts_key
        if rec_filter_func is None:
            rec_filter_func = self._rec_filter_func

        if isinstance(artifacts_key, str):
            artifacts_key = [artifacts_key]

        collect_dict = {}
        # filter records
        recs_flt = list_recorders(self.exp_name, rec_filter_func)
        for _, rec in recs_flt.items():
            rec_key = self._rec_key_func(rec)
            for key in artifacts_key:
                if self.ART_KEY_RAW == key:
                    artifact = rec
                else:
                    artifact = rec.load_object(self.artifacts_path[key])
                collect_dict.setdefault(key, {})[rec_key] = artifact

        return collect_dict
