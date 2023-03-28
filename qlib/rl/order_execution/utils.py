# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd

from qlib.backtest.decision import OrderDir
from qlib.backtest.executor import BaseExecutor, NestedExecutor, SimulatorExecutor
from qlib.constant import float_or_ndarray


def dataframe_append(df: pd.DataFrame, other: Any) -> pd.DataFrame:
    # dataframe.append is deprecated
    other_df = pd.DataFrame(other).set_index("datetime")
    other_df.index.name = "datetime"

    res = pd.concat([df, other_df], axis=0)
    return res


def price_advantage(
    exec_price: float_or_ndarray,
    baseline_price: float,
    direction: OrderDir | int,
) -> float_or_ndarray:
    if baseline_price == 0:  # something is wrong with data. Should be nan here
        if isinstance(exec_price, float):
            return 0.0
        else:
            return np.zeros_like(exec_price)
    if direction == OrderDir.BUY:
        res = (1 - exec_price / baseline_price) * 10000
    elif direction == OrderDir.SELL:
        res = (exec_price / baseline_price - 1) * 10000
    else:
        raise ValueError(f"Unexpected order direction: {direction}")
    res_wo_nan: np.ndarray = np.nan_to_num(res, nan=0.0)
    if res_wo_nan.size == 1:
        return res_wo_nan.item()
    else:
        return cast(float_or_ndarray, res_wo_nan)


def get_simulator_executor(executor: BaseExecutor) -> SimulatorExecutor:
    while isinstance(executor, NestedExecutor):
        executor = executor.inner_executor
    assert isinstance(executor, SimulatorExecutor)
    return executor
