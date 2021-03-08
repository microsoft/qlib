# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch.nn as nn

def count_parameters(models_or_parameters, unit="mb"):
    if isinstance(models_or_parameters, nn.Module):
        counts = sum(v.numel() for v in models_or_parameters.parameters())
    elif isinstance(models_or_parameters, nn.Parameter):
        counts = models_or_parameters.numel()
    elif isinstance(models_or_parameters, (list, tuple)):
        return sum(count_parameters(x, unit) for x in models_or_parameters)
    else:
        counts = sum(v.numel() for v in models_or_parameters)
    if unit.lower() == "mb":
        counts /= 1e6
    elif unit.lower() == "kb":
        counts /= 1e3
    elif unit.lower() == "gb":
        counts /= 1e9
    elif unit is not None:
        raise ValueError("Unknow unit: {:}".format(unit))
    return counts
