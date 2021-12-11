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

    def reweight(self, data: object) -> object:
        """
        Get weights for data

        Parameters
        ----------
        data : object
            The input data.
            The first dimension is the index of samples

        Returns
        -------
        object:
            the weights info for the data
        """
        raise NotImplementedError(f"This type of input is not supported")
