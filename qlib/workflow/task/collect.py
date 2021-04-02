from abc import abstractmethod
from typing import Callable, Union
from qlib.workflow.task.utils import list_recorders


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
    def __init__(
        self, exp_name, artifacts_path={"pred": "pred.pkl", "IC": "sig_analysis/ic.pkl"}, rec_key_func=None
    ) -> None:
        """init RecorderCollector

        Args:
            exp_name (str): the name of Experiment
            artifacts_path (dict, optional): The artifacts name and its path in Recorder. Defaults to {"pred": "pred.pkl", "IC": "sig_analysis/ic.pkl"}.
            rec_key_func (Callable): a function to get the key of a recorder. If None, use recorder id.
        """
        self.exp_name = exp_name
        self.artifacts_path = artifacts_path
        if rec_key_func is None:
            rec_key_func = lambda rec: rec.info["id"]
        self._get_key = rec_key_func

    def collect(self, artifacts_key=None, rec_filter_func=None):  # ensemble, get_group_key_func,
        """Collect different artifacts based on recorder after filtering.

        Args:
            artifacts_key (str or List, optional): the artifacts key you want to get. If None, get all artifacts.
            rec_filter_func (Callable, optional): filter the recorder by return True or False. Defaults to None.

        Returns:
            dict: the dict after collected like {artifact: {rec_key: object}}
        """
        if artifacts_key is None:
            artifacts_key = self.artifacts_path.keys()

        if isinstance(artifacts_key, str):
            artifacts_key = [artifacts_key]

        collect_dict = {}
        # filter records
        recs_flt = list_recorders(self.exp_name, rec_filter_func)
        for _, rec in recs_flt.items():
            rec_key = self._get_key(rec)
            for key in artifacts_key:
                artifact = rec.load_object(self.artifacts_path[key])
                collect_dict.setdefault(key, {})[rec_key] = artifact

        return collect_dict