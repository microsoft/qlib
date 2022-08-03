# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import TypeVar

import numpy as np
import pandas as pd

FINEST_GRANULARITY = "1min"
COARSEST_GRANULARITY = "1day"
ONE_SEC = pd.Timedelta("1s")  # use 1 second to exclude the right interval point
float_or_ndarray = TypeVar("float_or_ndarray", float, np.ndarray)