# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn


def get_device(GPU=0):
    """
    Select the appropriate device for PyTorch operations.

    Parameters
    ----------
    GPU : int or str
        If int: GPU device ID (>= 0 to use GPU if available, < 0 to force CPU)
        If str: Device string (e.g., "cuda:0", "mps", "cpu")

    Returns
    -------
    torch.device
        The selected device object

    Examples
    --------
    >>> device = get_device(0)  # Uses CUDA:0 if available, else MPS if available, else CPU
    >>> device = get_device("mps")  # Explicitly use MPS
    >>> device = get_device(-1)  # Force CPU
    """
    if isinstance(GPU, str):
        return torch.device(GPU)

    # If GPU is an int
    # GPU >= 0 means try to use GPU (CUDA or MPS), GPU < 0 means force CPU
    if GPU >= 0 and torch.cuda.is_available():
        return torch.device(f"cuda:{GPU}")
    elif GPU >= 0 and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def count_parameters(models_or_parameters, unit="m"):
    """
    This function is to obtain the storage size unit of a (or multiple) models.

    Parameters
    ----------
    models_or_parameters : PyTorch model(s) or a list of parameters.
    unit : the storage size unit.

    Returns
    -------
    The number of parameters of the given model(s) or parameters.
    """
    if isinstance(models_or_parameters, nn.Module):
        counts = sum(v.numel() for v in models_or_parameters.parameters())
    elif isinstance(models_or_parameters, nn.Parameter):
        counts = models_or_parameters.numel()
    elif isinstance(models_or_parameters, (list, tuple)):
        return sum(count_parameters(x, unit) for x in models_or_parameters)
    else:
        counts = sum(v.numel() for v in models_or_parameters)
    unit = unit.lower()
    if unit in ("kb", "k"):
        counts /= 2**10
    elif unit in ("mb", "m"):
        counts /= 2**20
    elif unit in ("gb", "g"):
        counts /= 2**30
    elif unit is not None:
        raise ValueError("Unknown unit: {:}".format(unit))
    return counts
