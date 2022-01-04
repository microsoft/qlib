# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

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
    if unit == "kb" or unit == "k":
        counts /= 2 ** 10
    elif unit == "mb" or unit == "m":
        counts /= 2 ** 20
    elif unit == "gb" or unit == "g":
        counts /= 2 ** 30
    elif unit is not None:
        raise ValueError("Unknown unit: {:}".format(unit))
    return counts
