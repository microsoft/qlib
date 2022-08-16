# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# REGION CONST
from typing import TypeVar

import numpy as np
import pandas as pd

REG_CN = "cn"
REG_US = "us"
REG_TW = "tw"

# Epsilon for avoiding division by zero.
EPS = 1e-12

# Infinity in integer
INF = 10**18
FINEST_GRANULARITY = "1min"
COARSEST_GRANULARITY = "1day"
ONE_SEC = pd.Timedelta("1s")  # use 1 second to exclude the right interval point
float_or_ndarray = TypeVar("float_or_ndarray", float, np.ndarray)