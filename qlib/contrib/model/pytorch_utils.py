# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import torch.nn as nn


def count_parameters(model_or_parameters, unit="mb"):
    if isinstance(model_or_parameters, nn.Module):
        counts = np.sum(np.prod(v.size()) for v in model_or_parameters.parameters())
    else:
        counts = np.sum(np.prod(v.size()) for v in model_or_parameters)
    if unit.lower() == "mb":
        counts /= 1e6
    elif unit.lower() == "kb":
        counts /= 1e3
    elif unit.lower() == "gb":
        counts /= 1e9
    elif unit is not None:
        raise ValueError("Unknow unit: {:}".format(unit))
    return counts
