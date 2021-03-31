from abc import abstractmethod
from typing import Callable, Union

import pandas as pd
from qlib import get_module_logger
from qlib.workflow.task.utils import list_recorders


class Collector:
    """The collector to collect different results based on experiment backend and ensemble method"""

    def collect(self, ensemble, get_group_key_func, *args, **kwargs):
        """To collect the results, we need to get the experiment record firstly and divided them into
        different groups. Then use ensemble methods to merge the group.

        Args:
            ensemble (Ensemble): an instance of Ensemble
            get_group_key_func (Callable): a function to get the group of a experiment record

        """
        raise NotImplementedError(f"Please implement the `collect` method.")


class RecorderCollector(Collector):
    def __init__(self, exp_name, artifacts_path={"pred": "pred.pkl", "IC": "sig_analysis/ic.pkl"}) -> None:
        """init RecorderCollector

        Args:
            exp_name (str): the name of Experiment
            artifacts_path (dict, optional): The artifacts name and its path in Recorder. Defaults to {"pred": "pred.pkl", "IC": "sig_analysis/ic.pkl"}.
        """
        self.exp_name = exp_name
        self.artifacts_path = artifacts_path

    def collect(self, ensemble, get_group_key_func, artifacts_key=None, rec_filter_func=None):
        """Collect different artifacts based on recorder after filtering and ensemble method.
        Group recorder by get_group_key_func.

        Args:
            ensemble (Ensemble): an instance of Ensemble
            get_group_key_func (Callable): a function to get the group of a experiment record
            artifacts_key (str or List, optional): the artifacts key you want to get. Defaults to None.
            rec_filter_func (Callable, optional): filter the recorder by return True or False. Defaults to None.

        Returns:
            dict: the dict after collected.
        """
        if artifacts_key is None:
            artifacts_key = self.artifacts_path.keys()

        if isinstance(artifacts_key, str):
            artifacts_key = [artifacts_key]

        # prepare_ensemble
        ensemble_dict = {}
        for key in artifacts_key:
            ensemble_dict.setdefault(key, {})
        # filter records
        recs_flt = list_recorders(self.exp_name, rec_filter_func)
        for _, rec in recs_flt.items():
            group_key = get_group_key_func(rec)
            for key in artifacts_key:
                artifact = rec.load_object(self.artifacts_path[key])
                ensemble_dict[key][group_key] = artifact

        if isinstance(artifacts_key, str):
            return ensemble(ensemble_dict[artifacts_key])

        collect_dict = {}
        for key in artifacts_key:
            collect_dict[key] = ensemble(ensemble_dict[key])
        return collect_dict
