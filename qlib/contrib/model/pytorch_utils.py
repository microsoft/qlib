# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn


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


def get_device(GPU=0, return_str=False):
    """
    Get the appropriate device (CUDA, MPS, or CPU) based on availability.

    Parameters
    ----------
    GPU : int
        the GPU ID used for training. If >= 0 and CUDA is available, use CUDA.
    return_str : bool
        if True, return device as string; if False, return torch.device object.

    Returns
    -------
    torch.device or str
        The device to use for computation.
    """
    USE_CUDA = torch.cuda.is_available() and GPU >= 0
    USE_MPS = torch.backends.mps.is_available()

    # Default to CPU, then check for GPU availability
    device_str = "cpu"
    if USE_CUDA:
        device_str = f"cuda:{GPU}"
    elif USE_MPS:
        device_str = "mps"

    if return_str:
        return device_str
    else:
        return torch.device(device_str)
