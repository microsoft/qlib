# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
    This module is not a necessary part of Qlib.
    They are just some tools for convenience
    It is should not imported into the core part of qlib
"""
import torch
import numpy as np
import pandas as pd


def data_to_tensor(data, device="cpu", raise_error=False):
    if isinstance(data, torch.Tensor):
        if device == "cpu":
            return data.cpu()
        else:
            return data.to(device)
    if isinstance(data, (pd.DataFrame, pd.Series)):
        return data_to_tensor(torch.from_numpy(data.values).float(), device)
    elif isinstance(data, np.ndarray):
        return data_to_tensor(torch.from_numpy(data).float(), device)
    elif isinstance(data, (tuple, list)):
        return [data_to_tensor(i, device) for i in data]
    elif isinstance(data, dict):
        return {k: data_to_tensor(v, device) for k, v in data.items()}
    else:
        if raise_error:
            raise ValueError(f"Unsupported data type: {type(data)}.")
        else:
            return data
