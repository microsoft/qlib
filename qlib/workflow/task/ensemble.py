from abc import abstractmethod
from typing import Callable, Union

import pandas as pd
from qlib import get_module_logger
from qlib.workflow.task.utils import list_recorders
from typing import Dict


class Ensemble:
    """Merge the objects in an Ensemble."""

    def __init__(self, merge_func=None, get_grouped_key_func=None) -> None:
        """init Ensemble

        Args:
            merge_func (Callable, optional): The specific merge function. Defaults to None.
            get_grouped_key_func (Callable, optional): Get group_inner_key and group_outer_key by group_key. Defaults to None.
        """
        self.logger = get_module_logger(self.__class__.__name__)
        if merge_func is not None:
            self.merge_func = merge_func
        if get_grouped_key_func is not None:
            self.get_grouped_key_func = get_grouped_key_func

    def merge_func(self, group_inner_dict):
        """Given a group_inner_dict such as {Rollinga_b: object, Rollingb_c: object},
        merge it to object

        Args:
            group_inner_dict (dict): the inner group dict

        """
        raise NotImplementedError(f"Please implement the `merge_func` method.")

    def get_grouped_key_func(self, group_key):
        """Given a group_key and return the group_outer_key, group_inner_key.

        For example:
            (A,B,Rolling) -> (A,B):Rolling
            (A,B) -> C:(A,B)

        Args:
            group_key (tuple or str): the group key
        """
        raise NotImplementedError(f"Please implement the `get_grouped_key_func` method.")

    def group(self, group_dict: Dict[tuple or str, object]) -> Dict[tuple or str, Dict[tuple or str, object]]:
        """In a group of dict, further divide them into outgroups and innergroup.

        For example:

        .. code-block:: python

            RollingEnsemble:
                input:
                {
                    (ModelA,Horizon5,Rollinga_b): object
                    (ModelA,Horizon5,Rollingb_c): object
                    (ModelA,Horizon10,Rollinga_b): object
                    (ModelA,Horizon10,Rollingb_c): object
                    (ModelB,Horizon5,Rollinga_b): object
                    (ModelB,Horizon5,Rollingb_c): object
                    (ModelB,Horizon10,Rollinga_b): object
                    (ModelB,Horizon10,Rollingb_c): object
                }

                output:
                {
                    (ModelA,Horizon5): {Rollinga_b: object, Rollingb_c: object}
                    (ModelA,Horizon10): {Rollinga_b: object, Rollingb_c: object}
                    (ModelB,Horizon5): {Rollinga_b: object, Rollingb_c: object}
                    (ModelB,Horizon10): {Rollinga_b: object, Rollingb_c: object}
                }

        Args:
            group_dict (Dict[tuple or str, object]): a group of dict

        Returns:
            Dict[tuple or str, Dict[tuple or str, object]]: the dict after `group`
        """
        grouped_dict = {}
        for group_key, artifact in group_dict.items():
            group_outer_key, group_inner_key = self.get_grouped_key_func(group_key)  # (A,B,Rolling) -> (A,B):Rolling
            grouped_dict.setdefault(group_outer_key, {})[group_inner_key] = artifact
        return grouped_dict

    def reduce(self, grouped_dict: dict):
        """After grouping, reduce the innergroup.

        For example:

        .. code-block:: python

            RollingEnsemble:
                input:
                {
                    (ModelA,Horizon5): {Rollinga_b: object, Rollingb_c: object}
                    (ModelA,Horizon10): {Rollinga_b: object, Rollingb_c: object}
                    (ModelB,Horizon5): {Rollinga_b: object, Rollingb_c: object}
                    (ModelB,Horizon10): {Rollinga_b: object, Rollingb_c: object}
                }

                output:
                {
                    (ModelA,Horizon5): object
                    (ModelA,Horizon10): object
                    (ModelB,Horizon5): object
                    (ModelB,Horizon10): object
                }

        Args:
            grouped_dict (dict): the dict after `group`

        Returns:
            dict: the dict after `reduce`
        """
        reduce_group = {}
        for group_outer_key, group_inner_dict in grouped_dict.items():
            artifact = self.merge_func(group_inner_dict)
            reduce_group[group_outer_key] = artifact
        return reduce_group

    def __call__(self, group_dict):
        """The process of Ensemble is group it firstly and then reduce it, like MapReduce.

        Args:
            group_dict (Dict[tuple or str, object]): a group of dict

        Returns:
            dict: the dict after `reduce`
        """
        grouped_dict = self.group(group_dict)
        return self.reduce(grouped_dict)


class RollingEnsemble(Ensemble):
    """A specific implementation of Ensemble for Rolling."""

    def merge_func(self, group_inner_dict):
        """merge group_inner_dict by datetime.

        Args:
            group_inner_dict (dict): the inner group dict

        Returns:
            object: the artifact after merging
        """
        artifact_list = list(group_inner_dict.values())
        artifact_list.sort(key=lambda x: x.index.get_level_values("datetime").min())
        artifact = pd.concat(artifact_list)
        # If there are duplicated predition, use the latest perdiction
        artifact = artifact[~artifact.index.duplicated(keep="last")]
        artifact = artifact.sort_index()
        return artifact

    def get_grouped_key_func(self, group_key):
        """The final axis of group_key must be the Rolling key.
        When `collect`, get_group_key_func can add the statement below.

        .. code-block:: python

            def get_group_key_func(recorder):
                task_config = recorder.load_object("task")
                ......
                rolling_key = task_config["dataset"]["kwargs"]["segments"]["test"]
            return ......, rolling_key

        Args:
            group_key (tuple or str): the group key

        Returns:
            tuple or str, tuple or str: group_outer_key, group_inner_key
        """
        assert len(group_key) >= 2
        return group_key[:-1], group_key[-1]
