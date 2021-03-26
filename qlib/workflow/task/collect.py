from abc import abstractmethod
from typing import Callable, Union

import pandas as pd
from qlib import get_module_logger
from qlib.workflow.task.utils import list_recorders


class Collector:
    """
    This class will divide disorderly records or anything worth collecting into different groups based on the group_key.
    After grouping, we can reduce the useful information from different groups.
    """

    def group(self, *args, **kwargs):
        """
        According to the get_group_key_func, divide disorderly things into different groups.

        For example:

        .. code-block:: python

            input:
            [thing1, thing2, thing3, thing4, thing5]

            output:
            {
                "group_name1": [thing3, thing5, thing1]
                "group_name2": [thing2, thing4]
            }

        Args:
            get_group_key_func (Callable): get a group key based on a thing
            things_list (list): a list of things

        Returns:
            dict: a dict including the group key and members of the group.

        """
        raise NotImplementedError(f"Please implement the `group` method.")

    def reduce(self, things_group: dict):
        """
        Using the dict from `group`, reduce useful information.

        Args:
            things_group (dict): a dict after grouping

        Returns:
            dict: a dict including the group key, the information key and the information value

        """
        raise NotImplementedError(f"Please implement the `reduce` method.")

    def collect(self, *args, **kwargs):
        """group and reduce

        Returns:
            dict: a dict including the group key, the information key and the information value
        """
        grouped = self.group(*args, **kwargs)
        return self.reduce(grouped)


class RecorderCollector(Collector):
    """
    The Recorder's Collector. This class is a implementation of Collector, collecting some artifacts saved by Recorder.
    """

    def __init__(self, experiment_name: str) -> None:
        self.exp_name = experiment_name
        self.logger = get_module_logger(self.__class__.__name__)

    _artifacts_key_path = {"pred": "pred.pkl", "IC": "sig_analysis/ic.pkl"}
    _artifacts_key_merge_method = {}

    def default_merge(self, artifact_list):
        """Merge disorderly artifacts in artifact list.

        Args:
            artifact_list (list): A artifact list.

        Raises:
            NotImplementedError: [description]
        """
        raise NotImplementedError(f"Please implement the `default_merge` method.")

    def group(self, get_group_key_func, rec_filter_func=None):
        """
        Filter recorders and group recorders by group key.

        Args:
            get_group_key_func (Callable): get a group key based on a recorder
            rec_filter_func (Callable, optional): filter the recorders in this experiment. Defaults to None.

        Returns:
            dict: a dict including the group key and recorders of the group
        """
        # filter records
        recs_flt = list_recorders(self.exp_name, rec_filter_func)

        # group
        recs_group = {}
        for _, rec in recs_flt.items():
            group_key = get_group_key_func(rec)
            recs_group.setdefault(group_key, []).append(rec)

        return recs_group

    def reduce(self, recs_group: dict, artifact_keys_list: list = None):
        """
        Reduce artifacts based on the dict of grouped recorder.
        The artifacts need be declared by artifact_keys_list.
        The artifacts path in recorder need be declared by _artifacts_key_path.
        If there is no declartion in _artifacts_key_merge_method, then use default_merge method to merge it.

        Args:
            recs_group (dict): The dict grouped by `group`
            artifact_keys_list (list): The list of artifact keys. If it is None, then use all artifacts in _artifacts_key_path.

        Returns:
            a dict including the group key, the artifact key and the artifact value.

            For example:

            .. code-block:: python

                {
                    group_key: {"pred": <VALUE>, "IC": <VALUE>}
                }
        """
        if artifact_keys_list == None:
            artifact_keys_list = self._artifacts_key_path.keys()
        reduce_group = {}
        for group_key, recorder_list in recs_group.items():
            reduced_artifacts = {}
            for artifact_key in artifact_keys_list:
                artifact_list = []
                for recorder in recorder_list:
                    artifact_list.append(recorder.load_object(self._artifacts_key_path[artifact_key]))
                merge_method = self._artifacts_key_merge_method.get(artifact_key, self.default_merge)
                artifact = merge_method(artifact_list)
                reduced_artifacts[artifact_key] = artifact
            reduce_group[group_key] = reduced_artifacts
        return reduce_group


class RollingCollector(RecorderCollector):
    """
    Collect the record results of the rolling tasks
    """

    def __init__(self, experiment_name: str):
        super().__init__(experiment_name)
        self.logger = get_module_logger(self.__class__.__name__)

    def default_merge(self, artifact_list):
        """merge disorderly artifacts based on the datetime.

        Args:
            artifact_list (list): a list of artifacts from different recorders

        Returns:
            merged artifact
        """
        # Make sure the pred are sorted according to the rolling start time
        artifact_list.sort(key=lambda x: x.index.get_level_values("datetime").min())
        artifact = pd.concat(artifact_list)
        # If there are duplicated predition, we use the latest perdiction
        artifact = artifact[~artifact.index.duplicated(keep="last")]
        artifact = artifact.sort_index()
        return artifact
