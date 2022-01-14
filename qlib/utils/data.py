# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from typing import Union
import pandas as pd
import numpy as np


def robust_zscore(x: pd.Series, zscore=False):
    """Robust ZScore Normalization

    Use robust statistics for Z-Score normalization:
        mean(x) = median(x)
        std(x) = MAD(x) * 1.4826

    Reference:
        https://en.wikipedia.org/wiki/Median_absolute_deviation.
    """
    x = x - x.median()
    mad = x.abs().median()
    x = np.clip(x / mad / 1.4826, -3, 3)
    if zscore:
        x -= x.mean()
        x /= x.std()
    return x


def zscore(x: Union[pd.Series, pd.DataFrame]):
    return (x - x.mean()).div(x.std())


def deepcopy_basic_type(obj: object) -> object:
    """
    deepcopy an object without copy the complicated objects.
        This is useful when you want to generate Qlib tasks and share the handler

    NOTE:
    - This function can't handle recursive objects!!!!!

    Parameters
    ----------
    obj : object
        the object to be copied

    Returns
    -------
    object:
        The copied object
    """
    if isinstance(obj, tuple):
        return tuple(deepcopy_basic_type(i) for i in obj)
    elif isinstance(obj, list):
        return list(deepcopy_basic_type(i) for i in obj)
    elif isinstance(obj, dict):
        return {k: deepcopy_basic_type(v) for k, v in obj.items()}
    else:
        return obj
