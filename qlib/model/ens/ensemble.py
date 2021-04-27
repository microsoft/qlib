from abc import abstractmethod
from typing import Callable, Union

import pandas as pd
from qlib.workflow.task.collect import Collector
from qlib.utils.serial import Serializable


def ens_workflow(collector: Collector, process_list, *args, **kwargs):
    """the ensemble workflow based on collector and different dict processors.

    Args:
        collector (Collector): the collector to collect the result into {result_key: things}
        process_list (list or Callable): the list of processors or the instance of processor to process dict.
        The processor order is same as the list order.
            For example: [Group1(..., Ensemble1()), Group2(..., Ensemble2())]
    Returns:
        dict: the ensemble dict
    """
    collect_dict = collector.collect()
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

    def __call__(self, ensemble_dict: dict, *args, **kwargs):
        """Merge the ensemble_dict into an ensemble object.
        For example: {Rollinga_b: object, Rollingb_c: object} -> object

        Args:
            ensemble_dict (dict): the ensemble dict waiting for merging like {name: things}

        Returns:
            object: the ensemble object
        """
        raise NotImplementedError(f"Please implement the `__call__` method.")


class RollingEnsemble(Ensemble):

    """Merge the rolling objects in an Ensemble"""

    def __call__(self, ensemble_dict: dict):
        """Merge a dict of rolling dataframe like `prediction` or `IC` into an ensemble.

        NOTE: The values of dict must be pd.Dataframe, and have the index "datetime"

        Args:
            ensemble_dict (dict): a dict like {"A": pd.Dataframe, "B": pd.Dataframe}.
            The key of the dict will be ignored.

        Returns:
            pd.Dataframe: the complete result of rolling.
        """
        artifact_list = list(ensemble_dict.values())
        artifact_list.sort(key=lambda x: x.index.get_level_values("datetime").min())
        artifact = pd.concat(artifact_list)
        # If there are duplicated predition, use the latest perdiction
        artifact = artifact[~artifact.index.duplicated(keep="last")]
        artifact = artifact.sort_index()
        return artifact
