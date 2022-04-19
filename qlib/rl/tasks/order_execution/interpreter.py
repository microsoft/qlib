# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict, cast

import numpy as np
import pandas as pd
from gym import spaces

from qlib.constant import EPS
from qlib.rl.interpreter import StateInterpreter, ActionInterpreter
from qlib.rl.tasks.data import pickle_styled

from .simulator_simple import SingleAssetOrderExecutionState


def canonicalize(value: int | float | np.ndarray | pd.DataFrame | dict) -> np.ndarray | dict:
    """To 32-bit numeric types. Recursively."""
    if isinstance(value, pd.DataFrame):
        return value.to_numpy()
    if isinstance(value, (float, np.floating)) or (
        isinstance(value, np.ndarray) and value.dtype.kind == 'f'
    ):
        return np.array(value, dtype=np.float32)
    elif isinstance(value, (int, bool, np.integer)) or (
        isinstance(value, np.ndarray) and value.dtype.kind == 'i'
    ):
        return np.array(value, dtype=np.int32)
    elif isinstance(value, dict):
        return {k: canonicalize(v) for k, v in value.items()}
    else:
        return value

class FullHistoryObsType(TypedDict):
    data_processed: np.ndarray
    data_processed_prev: np.ndarray
    acquiring: bool
    cur_time: int
    cur_step: int
    num_step: int
    target: float
    position: float
    position_history: np.ndarray


class FullHistoryStateInterpreter(StateInterpreter[FullHistoryObsType]):
    """The observation of all the history, including today (until this moment), and yesterday.

    Parameters
    ----------
    data_dir
        Path to load data after feature engineering.
    """

    def __init__(self, data_dir: Path, max_step: int, total_time: int, data_dim: int) -> None:
        self.data_dir = data_dir
        self.max_step = max_step
        self.total_time = total_time
        self.data_dim = data_dim

    def interpret(self, state: SingleAssetOrderExecutionState) -> FullHistoryObsType:
        processed = pickle_styled.get_intraday_processed_data(
            self.data_dir, state.order.stock_id, pd.Timestamp(state.order.start_time.date),
            self.data_dim, state.backtest_data.get_time_index()
        )

        return cast(FullHistoryObsType, canonicalize({
            'data_processed': self._mask_future_info(processed.today, state.cur_time),
            'data_processed_prev': processed.yesterday,
            'acquiring': state.order.direction == state.order.BUY,
            'cur_time': state.cur_time,
            'cur_step': self.env().status.cur_step,
            'num_step': self.max_step,
            'target': state.order.amount,
            'position': state.position,
            'position_history': np.array(state.position_history),
        }))

    @property
    def observation_space(self):
        space = {
            'data_processed': spaces.Box(-np.inf, np.inf, shape=(self.total_time, self.data_dim)),
            'data_processed_prev': spaces.Box(-np.inf, np.inf, shape=(self.total_time, self.data_dim)),
            'acquiring': spaces.Discrete(2),
            "cur_time": spaces.Box(0, self.total_time - 1, shape=(), dtype=np.int32),
            'cur_step': spaces.Box(0, self.max_step - 1, shape=(), dtype=np.int32),
            'num_step': spaces.Box(self.max_step, self.max_step, shape=(), dtype=np.int32),
            'target': spaces.Box(-EPS, np.inf, shape=()),
            'position': spaces.Box(-EPS, np.inf, shape=()),
            'position_history': spaces.Box(-EPS, np.inf, shape=(self.max_step,)),
        }
        return spaces.Dict(space)

    @staticmethod
    def _mask_future_info(arr: pd.DataFrame, current: pd.Timestamp):
        arr = arr.copy(deep=True)
        arr.loc[current:] = 0.  # mask out data after this moment (inclusive)
        return arr


class CurrentStepObservation(StateInterpreter):
    def __init__(self, max_step: int):
        self.max_step = max_step

    @property
    def observation_space(self):
        space = {
            'acquiring': spaces.Discrete(2),
            'cur_step': spaces.Box(0, self.max_step - 1, shape=(), dtype=np.int32),
            'num_step': spaces.Box(self.max_step, self.max_step, shape=(), dtype=np.int32),
            'target': spaces.Box(-EPS, np.inf, shape=()),
            'position': spaces.Box(-EPS, np.inf, shape=()),
        }
        return spaces.Dict(space)

    def interpret(self, state: SingleAssetOrderExecutionState) -> Any:
        assert self.env().status.cur_step <= self.max_step
        obs = {
            'acquiring': state.order.direction == state.order.BUY,
            'cur_step': self.env().status.cur_step,
            'num_step': self.max_step,
            'target': state.order.amount,
            'position': state.position,
        }
        return obs


class CategoricalAction(ActionInterpreter):
    def __init__(self, values: int | list[float]):
        if isinstance(values, int):
            values = [i / values for i in range(0, values + 1)]
        self.action_values = values

    @property
    def action_space(self):
        return spaces.Discrete(len(self.action_values))

    def to_volume(self, state: SingleAssetOrderExecutionState, action: int) -> float:
        assert 0 <= action < len(self.action_values)
        return min(state.position, state.order.amount * self.action_values[action])


class TwapRemainingAdjustmentWOSplit(ActionInterpreter):
    @property
    def action_space(self):
        return spaces.Box(0, np.inf, shape=(), dtype=np.float32)

    def to_volume(self, state: SingleAssetOrderExecutionState, action: float) -> float:
        twap_volume = state.order.position / (state.order.estimated_total_steps - self.env().status.cur_step)
        return min(state.position, twap_volume * action)
