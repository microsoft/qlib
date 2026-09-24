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
INF = int(1e18)
# pd.Timedelta("1day")-style string parsing hits pandas's deprecated
# "generic" NumPy timedelta unit path and warns on every import of this
# module; pd.Timedelta(<n>, <unit>) takes the same explicit-unit path
# used internally to build a concrete timedelta64[ns] value without it.
ONE_DAY = pd.Timedelta(1, "D")
ONE_MIN = pd.Timedelta(1, "min")
EPS_T = pd.Timedelta(1, "s")  # use 1 second to exclude the right interval point
float_or_ndarray = TypeVar("float_or_ndarray", float, np.ndarray)
