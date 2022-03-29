from __future__ import annotations

from typing import Any

import numpy as np
from gym import spaces

from qlib.rl.config import STATE_INTERPRETERS, EPSILON
from qlib.rl.interpreter import StateInterpreter

from .simulator_simple import SingleAssetOrderExecutionState


def _n32(value: int | float | np.ndarray | dict) -> np.ndarray | dict:
    """To 32-bit numeric types. Recursively."""
    if isinstance(value, (float, np.floating)) or (
        isinstance(value, np.ndarray) and value.dtype.kind == 'f'
    ):
        return np.array(value, dtype=np.float32)
    elif isinstance(value, (int, bool, np.integer)) or (
        isinstance(value, np.ndarray) and value.dtype.kind == 'i'
    ):
        return np.array(value, dtype=np.int32)
    elif isinstance(value, dict):
        return {k: _n32(v) for k, v in value.items()}
    else:
        return value


@STATE_INTERPRETERS.register_module('saoe_full_history')
class FullHistoryStateInterpreter(StateInterpreter):

    def __init__(self, max_step: int, total_time: int, data_dim: int) -> None:
        self.max_step = max_step
        self.total_time = total_time
        self.data_dim = data_dim
        self._feature_loader = []

    def interpret(self, state: SingleAssetOrderExecutionState) -> Any:
        return _n32({
            'data_processed': self._mask_future_info(state.get_processed_data(), state.cur_time),
            'data_processed_prev': state.get_processed_data('yesterday'),
            'acquiring': state.order.direction == state.order.BUY,
            'cur_time': min(state.cur_time, state.end_time - 1),
            'cur_step': self.env().status.cur_step,
            'num_step': self.max_step,
            'target': state.order.amount,
            'position': state.position,
            'position_history': np.array(state.position_history),
        })

    @property
    def observation_space(self):
        space = {
            'data_processed': spaces.Box(-np.inf, np.inf, shape=(self.total_time, self.data_dim)),
            'data_processed_prev': spaces.Box(-np.inf, np.inf, shape=(self.total_time, self.data_dim)),
            'acquiring': spaces.Discrete(2),
            "cur_time": spaces.Box(0, self.total_time - 1, shape=(), dtype=np.int32),
            'cur_step': spaces.Box(0, self.max_step - 1, shape=(), dtype=np.int32),
            'num_step': spaces.Box(self.max_step, self.max_step, shape=(), dtype=np.int32),
            'target': spaces.Box(-EPSILON, np.inf, shape=()),
            'position': spaces.Box(-EPSILON, np.inf, shape=()),
            'position_history': spaces.Box(-EPSILON, np.inf, shape=(self.max_step,)),
        }
        return spaces.Dict(space)

    @staticmethod
    def _mask_future_info(arr, current):
        arr = arr.copy()
        arr[current:] = 0.
        return arr


@STATE_INTERPRETERS.register_module('current_step')
class CurrentStepObservation(StateInterpreter):
    def __init__(self, max_step: int):
        self.max_step = max_step

    @property
    def observation_space(self):
        space = {
            'acquiring': spaces.Discrete(2),
            'cur_step': spaces.Box(0, self.max_step - 1, shape=(), dtype=np.int32),
            'num_step': spaces.Box(self.max_step, self.max_step, shape=(), dtype=np.int32),
            'target': spaces.Box(-EPSILON, np.inf, shape=()),
            'position': spaces.Box(-EPSILON, np.inf, shape=()),
        }
        return spaces.Dict(space)

    def interpret(self, state: SingleAssetOrderExecutionState) -> Any:
        assert self.env().status.cur_step <= self.num_step
        obs = {
            'acquiring': state.order.direction == state.order.BUY,
            'cur_step': self.env().status.cur_step,
            'num_step': self.max_step,
            'target': state.order.amount,
            'position': state.position,
        }
        return obs
