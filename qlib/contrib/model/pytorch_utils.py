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


def get_torch_device(GPU=0):
    """Select the best available torch device.

    Priority: CUDA > MPS (Apple Silicon) > CPU.

    Parameters
    ----------
    GPU : int or str
        If int >= 0 and CUDA is available, use ``cuda:<GPU>``.
        If str, use the value directly (e.g. ``"cuda:1"``).
        If int < 0, skip CUDA and fall through to MPS/CPU.

    Returns
    -------
    torch.device
    """
    if isinstance(GPU, str):
        return torch.device(GPU)
    if torch.cuda.is_available() and GPU >= 0:
        return torch.device(f"cuda:{GPU}")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
