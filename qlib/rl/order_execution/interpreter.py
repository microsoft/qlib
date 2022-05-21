# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from gym import spaces

from qlib.constant import EPS
from qlib.rl.interpreter import StateInterpreter, ActionInterpreter
from qlib.rl.data import pickle_styled
from qlib.typehint import TypedDict

from .simulator_simple import SAOEState

__all__ = [
    "FullHistoryStateInterpreter",
    "CurrentStepStateInterpreter",
    "CategoricalActionInterpreter",
    "TwapRelativeActionInterpreter",
]


def canonicalize(value: int | float | np.ndarray | pd.DataFrame | dict) -> np.ndarray | dict:
    """To 32-bit numeric types. Recursively."""
    if isinstance(value, pd.DataFrame):
        return value.to_numpy()
    if isinstance(value, (float, np.floating)) or (isinstance(value, np.ndarray) and value.dtype.kind == "f"):
        return np.array(value, dtype=np.float32)
    elif isinstance(value, (int, bool, np.integer)) or (isinstance(value, np.ndarray) and value.dtype.kind == "i"):
        return np.array(value, dtype=np.int32)
    elif isinstance(value, dict):
        return {k: canonicalize(v) for k, v in value.items()}
    else:
        return value


class FullHistoryObs(TypedDict):
    data_processed: Any
    data_processed_prev: Any
    acquiring: Any
    cur_tick: Any
    cur_step: Any
    num_step: Any
    target: Any
    position: Any
    position_history: Any


class FullHistoryStateInterpreter(StateInterpreter[SAOEState, FullHistoryObs]):
    """The observation of all the history, including today (until this moment), and yesterday.

    Parameters
    ----------
    data_dir
        Path to load data after feature engineering.
    max_step
        Total number of steps (an upper-bound estimation). For example, 390min / 30min-per-step = 13 steps.
    data_ticks
        Equal to the total number of records. For example, in SAOE per minute,
        the total ticks is the length of day in minutes.
    data_dim
        Number of dimensions in data.
    """

    def __init__(self, data_dir: Path, max_step: int, data_ticks: int, data_dim: int) -> None:
        self.data_dir = data_dir
        self.max_step = max_step
        self.data_ticks = data_ticks
        self.data_dim = data_dim

    def interpret(self, state: SAOEState) -> FullHistoryObs:
        processed = pickle_styled.load_intraday_processed_data(
            self.data_dir,
            state.order.stock_id,
            pd.Timestamp(state.order.start_time.date()),
            self.data_dim,
            state.ticks_index,
        )

        position_history = np.full(self.max_step + 1, 0.0, dtype=np.float32)
        position_history[0] = state.order.amount
        position_history[1 : len(state.history_steps) + 1] = state.history_steps["position"].to_numpy()

        assert self.env is not None

        # The min, slice here are to make sure that indices fit into the range,
        # even after the final step of the simulator (in the done step),
        # to make network in policy happy.
        return cast(
            FullHistoryObs,
            canonicalize(
                {
                    "data_processed": self._mask_future_info(processed.today, state.cur_time),
                    "data_processed_prev": processed.yesterday,
                    "acquiring": state.order.direction == state.order.BUY,
                    "cur_tick": min(np.sum(state.ticks_index < state.cur_time), self.data_ticks - 1),
                    "cur_step": min(self.env.status["cur_step"], self.max_step - 1),
                    "num_step": self.max_step,
                    "target": state.order.amount,
                    "position": state.position,
                    "position_history": position_history[: self.max_step],
                }
            ),
        )

    @property
    def observation_space(self):
        space = {
            "data_processed": spaces.Box(-np.inf, np.inf, shape=(self.data_ticks, self.data_dim)),
            "data_processed_prev": spaces.Box(-np.inf, np.inf, shape=(self.data_ticks, self.data_dim)),
            "acquiring": spaces.Discrete(2),
            "cur_tick": spaces.Box(0, self.data_ticks - 1, shape=(), dtype=np.int32),
            "cur_step": spaces.Box(0, self.max_step - 1, shape=(), dtype=np.int32),
            # TODO: support arbitrary length index
            "num_step": spaces.Box(self.max_step, self.max_step, shape=(), dtype=np.int32),
            "target": spaces.Box(-EPS, np.inf, shape=()),
            "position": spaces.Box(-EPS, np.inf, shape=()),
            "position_history": spaces.Box(-EPS, np.inf, shape=(self.max_step,)),
        }
        return spaces.Dict(space)

    @staticmethod
    def _mask_future_info(arr: pd.DataFrame, current: pd.Timestamp) -> pd.DataFrame:
        arr = arr.copy(deep=True)
        arr.loc[current:] = 0.0  # mask out data after this moment (inclusive)
        return arr


class CurrentStateObs(TypedDict):
    acquiring: bool
    cur_step: int
    num_step: int
    target: float
    position: float


class CurrentStepStateInterpreter(StateInterpreter[SAOEState, CurrentStateObs]):
    """The observation of current step.

    Used when policy only depends on the latest state, but not history.
    The key list is not full. You can add more if more information is needed by your policy.
    """

    def __init__(self, max_step: int):
        self.max_step = max_step

    @property
    def observation_space(self):
        space = {
            "acquiring": spaces.Discrete(2),
            "cur_step": spaces.Box(0, self.max_step - 1, shape=(), dtype=np.int32),
            "num_step": spaces.Box(self.max_step, self.max_step, shape=(), dtype=np.int32),
            "target": spaces.Box(-EPS, np.inf, shape=()),
            "position": spaces.Box(-EPS, np.inf, shape=()),
        }
        return spaces.Dict(space)

    def interpret(self, state: SAOEState) -> CurrentStateObs:
        assert self.env is not None
        assert self.env.status["cur_step"] <= self.max_step
        obs = CurrentStateObs(
            {
                "acquiring": state.order.direction == state.order.BUY,
                "cur_step": self.env.status["cur_step"],
                "num_step": self.max_step,
                "target": state.order.amount,
                "position": state.position,
            }
        )
        return obs


class CategoricalActionInterpreter(ActionInterpreter[SAOEState, int, float]):
    """Convert a discrete policy action to a continuous action, then multiplied by ``order.amount``.

    Parameters
    ----------
    values
        It can be a list of length $L$: $[a_1, a_2, \\ldots, a_L]$.
        Then when policy givens decision $x$, $a_x$ times order amount is the output.
        It can also be an integer $n$, in which case the list of length $n+1$ is auto-generated,
        i.e., $[0, 1/n, 2/n, \\ldots, n/n]$.
    """

    def __init__(self, values: int | list[float]):
        if isinstance(values, int):
            values = [i / values for i in range(0, values + 1)]
        self.action_values = values

    @property
    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_values))

    def interpret(self, state: SAOEState, action: int) -> float:
        assert 0 <= action < len(self.action_values)
        return min(state.position, state.order.amount * self.action_values[action])


class TwapRelativeActionInterpreter(ActionInterpreter[SAOEState, float, float]):
    """Convert a continous ratio to deal amount.

    The ratio is relative to TWAP on the remainder of the day.
    For example, there are 5 steps left, and the left position is 300.
    With TWAP strategy, in each position, 60 should be traded.
    When this interpreter receives action $a$, its output is $60 \\cdot a$.
    """

    @property
    def action_space(self) -> spaces.Box:
        return spaces.Box(0, np.inf, shape=(), dtype=np.float32)

    def interpret(self, state: SAOEState, action: float) -> float:
        assert self.env is not None
        estimated_total_steps = math.ceil(len(state.ticks_for_order) / state.ticks_per_step)
        twap_volume = state.position / (estimated_total_steps - self.env.status["cur_step"])
        return min(state.position, twap_volume * action)
