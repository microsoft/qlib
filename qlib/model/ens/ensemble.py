# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Ensemble can merge the objects in an Ensemble. For example, if there are many submodels predictions, we may need to merge them in an ensemble predictions.
"""

import pandas as pd


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
