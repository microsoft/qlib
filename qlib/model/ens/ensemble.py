# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Ensemble can merge the objects in an Ensemble. For example, if there are many submodels predictions, we may need to merge them in an ensemble predictions.
"""

from typing import Union
import pandas as pd
from qlib.utils import flatten_dict


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


class SingleKeyEnsemble(Ensemble):

    """
    Extract the object if there is only one key and value in dict. Make result more readable.
    {Only key: Only value} -> Only value
    If there are more than 1 key or less than 1 key, then do nothing.
    Even you can run this recursively to make dict more readable.
    NOTE: Default run recursively.
    """

    def __call__(self, ensemble_dict: Union[dict, object], recursion: bool = True) -> object:
        if not isinstance(ensemble_dict, dict):
            return ensemble_dict
        if recursion:
            tmp_dict = {}
            for k, v in ensemble_dict.items():
                tmp_dict[k] = self(v, recursion)
            ensemble_dict = tmp_dict
        keys = list(ensemble_dict.keys())
        if len(keys) == 1:
            ensemble_dict = ensemble_dict[keys[0]]
        return ensemble_dict


class RollingEnsemble(Ensemble):

    """Merge the rolling objects in an Ensemble"""

    def __call__(self, ensemble_dict: dict) -> pd.DataFrame:
        """Merge a dict of rolling dataframe like `prediction` or `IC` into an ensemble.

        NOTE: The values of dict must be pd.DataFrame, and have the index "datetime"

        Args:
            ensemble_dict (dict): a dict like {"A": pd.DataFrame, "B": pd.DataFrame}.
            The key of the dict will be ignored.

        Returns:
            pd.DataFrame: the complete result of rolling.
        """
        artifact_list = list(ensemble_dict.values())
        artifact_list.sort(key=lambda x: x.index.get_level_values("datetime").min())
        artifact = pd.concat(artifact_list)
        # If there are duplicated predition, use the latest perdiction
        artifact = artifact[~artifact.index.duplicated(keep="last")]
        artifact = artifact.sort_index()
        return artifact


class AverageEnsemble(Ensemble):
    def __call__(self, ensemble_dict: dict):
        """
        Average and standardize a dict of same shape dataframe like `prediction` or `IC` into an ensemble.

        NOTE: The values of dict must be pd.DataFrame, and have the index "datetime". If it is a nested dict, then flat it.

        Args:
            ensemble_dict (dict): a dict like {"A": pd.DataFrame, "B": pd.DataFrame}.
            The key of the dict will be ignored.

        Returns:
            pd.DataFrame: the complete result of averaging and standardizing.
        """
        # need to flatten the nested dict
        ensemble_dict = flatten_dict(ensemble_dict)
        values = list(ensemble_dict.values())
        results = pd.concat(values, axis=1)
        results = results.groupby("datetime").apply(lambda df: (df - df.mean()) / df.std())
        results = results.mean(axis=1)
        results = results.sort_index()
        return results
