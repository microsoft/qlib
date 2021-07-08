# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pandas as pd
import numpy as np
from typing import Union, List, Tuple
from ...data.dataset import TSDataSampler
from ...data.dataset.utils import get_level_index
from ...utils import lazy_sort_index


class Reweighter:
    def __init__(self, *args, **kwargs):
        """
        To initialize the Reweighter, users should provide specific methods to let reweighter do the reweighting (such as sample-wise, rule-based).
        """
        raise NotImplementedError()


class WeightSampler:
    """
    (T)ime-(S)eries WeightSampler
    This is the result of the function prepare_weight.

    It is aligned with the instance of TSDataSampler.
    """

    def __init__(self, weights: pd.Series):
        assert get_level_index(weights, "datetime") == 0
        self.weights_s = lazy_sort_index(weights)

    def __getitem__(self, idx: int):
        return self.weights_s[idx]

    def __len__(self):
        return len(self.weights_s)


class SampleReweighter(Reweighter):
    """
    The sample-wise reweighter. It aims to reweight by the given weight of each sample.
    """

    def __init__(self, sample_weights: pd.Series, *args, **kwargs):
        """

        Parameters
        ----------
        sample_weights : pd.Series
            Determine the weight of each sample.
            The index of the Series should be exactly the same with each sample's index.
        """
        self.weights = sample_weights

    def _sample_reweight_DataFrame(self, samples: Union[pd.Series, pd.DataFrame], *args, **kwargs) -> pd.Series:
        """
        This function processes the prepared data with pd.Series or pd.DataFrame type.

        Returns
        -------
        pd.Series:
            The weights of the prepared data.
        """
        weight = pd.Series(data=1.0, index=samples.index, name="weight")
        weight.update(self.weights)
        return weight

    def _sample_reweight_TSDataSampler(self, sampler: TSDataSampler, *args, **kwargs):
        """
        This function processes the prepared data with TSDataSampler type.

        Returns
        -------
        WeightSampler:
            The weight sampler of the prepared data.
        """
        weight = pd.Series(1.0, index=sampler.get_index(), name="weight")
        weight.update(self.weights)
        return WeightSampler(weight)

    def reweight(self, prepared_data: Union[list, tuple, pd.DataFrame, pd.Series, WeightSampler]):
        """
        Reweight the prepared data.

        Parameters
        ----------
        prepared_data: Union[list, tuple, pd.DataFrame, pd.Series, WeightSampler]
            The prepared data given by the DatasetH.

        Returns
        -------
        Union[list, pd.Series, WeightSampler]:
        """
        # Handle all kinds of prepared data format
        if isinstance(prepared_data, (list, tuple)):
            return [self.reweight(data) for data in prepared_data]
        elif isinstance(prepared_data, (pd.Series, pd.DataFrame)):
            return self._sample_reweight_DataFrame(prepared_data)
        elif isinstance(prepared_data, TSDataSampler):
            return self._sample_reweight_TSDataSampler(prepared_data)
        else:
            raise NotImplementedError(f"This type of input is not supported")
