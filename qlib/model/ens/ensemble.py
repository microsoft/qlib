from abc import abstractmethod
from typing import Callable, Union

import pandas as pd
from qlib.workflow.task.collect import Collector


def ens_workflow(collector: Collector, process_list, artifacts_key=None, rec_filter_func=None, *args, **kwargs):
    """the ensemble workflow based on collector and different dict processors.

    Args:
        collector (Collector): the collector to collect the result into {result_key: things}
        process_list (list or Callable): the list of processors or the instance of processor to process dict.
        The processor order is same as the list order.

        For example: [Group1(..., Ensemble1()), Group2(..., Ensemble2())]

        artifacts_key (list, optional): the artifacts key you want to get. If None, get all artifacts.
        rec_filter_func (Callable, optional): filter the recorder by return True or False. Defaults to None.

    Returns:
        dict: the ensemble dict
    """
    collect_dict = collector.collect(artifacts_key=artifacts_key, rec_filter_func=rec_filter_func)
    if not isinstance(process_list, list):
        process_list = [process_list]

    ensemble = {}
    for artifact in collect_dict:
        value = collect_dict[artifact]
        for process in process_list:
            if not callable(process):
                raise NotImplementedError(f"{type(process)} is not supported in `ens_workflow`.")
            value = process(value, *args, **kwargs)
        ensemble[artifact] = value

    return ensemble


class Ensemble:
    """Merge the objects in an Ensemble."""

    def __init__(self, merge_func=None):
        """init Ensemble

        Args:
            merge_func (Callable, optional): Given a dict and return the ensemble.

                For example: {Rollinga_b: object, Rollingb_c: object} -> object

            Defaults to None.
        """
        self._merge = merge_func

    def __call__(self, ensemble_dict: dict, *args, **kwargs):
        """Merge the ensemble_dict into an ensemble object.

        Args:
            ensemble_dict (dict): the ensemble dict waiting for merging like {name: things}

        Returns:
            object: the ensemble object
        """
        if isinstance(getattr(self, "_merge", None), Callable):
            return self._merge(ensemble_dict, *args, **kwargs)
        else:
            raise NotImplementedError(f"Please specify valid merge_func.")


class RollingEnsemble(Ensemble):

    """Merge the rolling objects in an Ensemble"""

    @staticmethod
    def rolling_merge(rolling_dict: dict):
        """Merge a dict of rolling dataframe like `prediction` or `IC` into an ensemble.

        NOTE: The values of dict must be pd.Dataframe, and have the index "datetime"

        Args:
            rolling_dict (dict): a dict like {"A": pd.Dataframe, "B": pd.Dataframe}.
            The key of the dict will be ignored.

        Returns:
            pd.Dataframe: the complete result of rolling.
        """
        artifact_list = list(rolling_dict.values())
        artifact_list.sort(key=lambda x: x.index.get_level_values("datetime").min())
        artifact = pd.concat(artifact_list)
        # If there are duplicated predition, use the latest perdiction
        artifact = artifact[~artifact.index.duplicated(keep="last")]
        artifact = artifact.sort_index()
        return artifact

    def __init__(self, merge_func=None):
        super().__init__(merge_func=merge_func)
        if merge_func is None:
            self._merge = RollingEnsemble.rolling_merge