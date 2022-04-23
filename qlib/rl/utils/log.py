# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Distributed logger for RL.

:class:`LogCollector` runs in every environment workers. It collects log info from simulator states,
and add them (as a dict) to auxiliary info returned for each step.

:class:`LogWriter` runs in the central worker. It decodes the dict collected by :class:`LogCollector`
in each worker, and writes them to console, log files, or tensorboard...

The two modules communicate by the "log" field in "info" returned by ``env.step()``.
"""

from __future__ import annotations

import logging
from enum import IntEnum
from typing import Any, TypeVar, Generic, Set, TYPE_CHECKING

from qlib.log import get_module_logger

if TYPE_CHECKING:
    from .env_wrapper import InfoDict

import inspect
import json
from collections import defaultdict
from typing import TextIO

import numpy as np
import pandas as pd


__all__ = ["LogCollector", "LogWriter"]

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class LogLevel(IntEnum):
    """Log-levels for RL training.
    The behavior of handling each log level depends on the implementation of :class:`LogWriter`.
    """
    DEBUG = 10          # If you only want to see the metric in debug mode
    PERIODIC = 20       # If you want to see the metric periodically
    INFO = 30           # Important log messages
    CRITICAL = 40       # LogWriter should always handle CRITICAL messages


class LogCollector:
    """Logs are first collected in each environment worker,
    and then aggregated to stream at the central thread in vector env.
    
    In :class:`LogCollector`, every metric is added to a dict, which needs to be ``reset()`` at each step.
    The dict is sent via the ``info`` in ``env.step()``, and decoded by the :class:`LogWriter` at vector env.
    """

    _logged: dict[str, tuple[int, Any]]

    def reset(self):
        """Clear all collected contents."""
        self._logged = {}

    def add_scalar(self, name: str, scalar: Any, loglevel: int | LogLevel = LogLevel.PERIODIC) -> None:
        """Add a scalar with name into logged contents.
        Scalar will be converted into a float.
        """
        if hasattr(scalar, "item"):
            # could be single-item number
            scalar = scalar.item()
        if not isinstance(scalar, (float, int)):
            raise TypeError("{scalar} is not and can not be converted into float or integer.")
        scalar = float(scalar)
        if name in self._logged:
            raise ValueError(f"A metric with {name} is already added. Please change a name or reset the log collector.")
        self._logged[name] = (int(loglevel), scalar)

    def add_array(self, name: str, array: np.ndarray | pd.DataFrame | pd.Series,
                  loglevel: int | LogLevel = LogLevel.PERIODIC) -> None:
        """Add an array with name into logging."""
        # FIXME: check whether venv allows logs to have a dynamic key set.
        if not isinstance(array, (np.ndarray, pd.DataFrame, pd.Series)):
            raise TypeError("{array} is not one of ndarray, DataFrame and Series.")
        if name in self._logged:
            raise ValueError(f"A metric with {name} is already added. Please change a name or reset the log collector.")
        self._logged[name] = (int(loglevel), array)

    def logs(self) -> dict[str, tuple[int, Any]]:
        return self._logged


class LogWriter(Generic[ObsType, ActType]):
    """Base class for log writers, triggered at every reset and step by finite env."""

    ep_count: int
    """Counter of episodes."""

    step_count: int
    """Counter of steps."""

    global_step: int
    """Counter of steps. Won"t be cleared in ``clear``."""

    active_env_ids: Set[int]
    """Active environment ids in vector env."""

    episode_lengths: dict[int, int]
    """Map from environment id to episode length."""

    episode_rewards: dict[int, int]
    """Map from environment id to episode total reward."""

    def __init__(self):
        self.global_step = 0

        self.episode_lengths = dict()
        self.episode_rewards = dict()
        self.episode_rewards_info = dict()

        self.clear()

    def clear(self):
        self.ep_count = self.step_count = 0
        self.active_env_ids = set()
        self.logs = []

    def log(self, done: bool, contents: dict):
        print(done, contents)

    def on_env_step(self, env_id: int, obs: ObsType, rew: float, done: bool, info: InfoDict):
        self.active_env_ids.add(env_id)
        self.episode_lengths[env_id] += 1
        self.episode_rewards[env_id] += rew

        self.log(done, info["log"])

    def on_env_reset(self, env_id: int, obs: ObsType):
        """Reset episode statistics. Nothing task-specific is logged here because of
        `a limitation of tianshou <https://github.com/thu-ml/tianshou/issues/605>`__.
        """
        self.episode_lengths[env_id] = 0
        self.episode_rewards[env_id] = 0.0
        self.episode_rewards_info[env_id] = defaultdict(float)
