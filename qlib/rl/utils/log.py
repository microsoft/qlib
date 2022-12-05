# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Distributed logger for RL.

:class:`LogCollector` runs in every environment workers. It collects log info from simulator states,
and add them (as a dict) to auxiliary info returned for each step.

:class:`LogWriter` runs in the central worker. It decodes the dict collected by :class:`LogCollector`
in each worker, and writes them to console, log files, or tensorboard...

The two modules communicate by the "log" field in "info" returned by ``env.step()``.
"""

# NOTE: This file contains many hardcoded / ad-hoc rules.
# Refactoring it will be one of the future tasks.

from __future__ import annotations

import logging
from collections import defaultdict
from enum import IntEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Generic, List, Sequence, Set, Tuple, TypeVar

import numpy as np
import pandas as pd

from qlib.log import get_module_logger

if TYPE_CHECKING:
    from .env_wrapper import InfoDict


__all__ = ["LogCollector", "LogWriter", "LogLevel", "LogBuffer", "ConsoleWriter", "CsvWriter"]

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class LogLevel(IntEnum):
    """Log-levels for RL training.
    The behavior of handling each log level depends on the implementation of :class:`LogWriter`.
    """

    DEBUG = 10
    """If you only want to see the metric in debug mode."""
    PERIODIC = 20
    """If you want to see the metric periodically."""
    # FIXME: I haven't given much thought about this. Let's hold it for one iteration.

    INFO = 30
    """Important log messages."""
    CRITICAL = 40
    """LogWriter should always handle CRITICAL messages"""


class LogCollector:
    """Logs are first collected in each environment worker,
    and then aggregated to stream at the central thread in vector env.

    In :class:`LogCollector`, every metric is added to a dict, which needs to be ``reset()`` at each step.
    The dict is sent via the ``info`` in ``env.step()``, and decoded by the :class:`LogWriter` at vector env.

    ``min_loglevel`` is for optimization purposes: to avoid too much traffic on networks / in pipe.
    """

    _logged: Dict[str, Tuple[int, Any]]
    _min_loglevel: int

    def __init__(self, min_loglevel: int | LogLevel = LogLevel.PERIODIC) -> None:
        self._min_loglevel = int(min_loglevel)

    def reset(self) -> None:
        """Clear all collected contents."""
        self._logged = {}

    def _add_metric(self, name: str, metric: Any, loglevel: int | LogLevel) -> None:
        if name in self._logged:
            raise ValueError(f"A metric with {name} is already added. Please change a name or reset the log collector.")
        self._logged[name] = (int(loglevel), metric)

    def add_string(self, name: str, string: str, loglevel: int | LogLevel = LogLevel.PERIODIC) -> None:
        """Add a string with name into logged contents."""
        if loglevel < self._min_loglevel:
            return
        if not isinstance(string, str):
            raise TypeError(f"{string} is not a string.")
        self._add_metric(name, string, loglevel)

    def add_scalar(self, name: str, scalar: Any, loglevel: int | LogLevel = LogLevel.PERIODIC) -> None:
        """Add a scalar with name into logged contents.
        Scalar will be converted into a float.
        """
        if loglevel < self._min_loglevel:
            return

        if hasattr(scalar, "item"):
            # could be single-item number
            scalar = scalar.item()
        if not isinstance(scalar, (float, int)):
            raise TypeError(f"{scalar} is not and can not be converted into float or integer.")
        scalar = float(scalar)
        self._add_metric(name, scalar, loglevel)

    def add_array(
        self,
        name: str,
        array: np.ndarray | pd.DataFrame | pd.Series,
        loglevel: int | LogLevel = LogLevel.PERIODIC,
    ) -> None:
        """Add an array with name into logging."""
        if loglevel < self._min_loglevel:
            return

        if not isinstance(array, (np.ndarray, pd.DataFrame, pd.Series)):
            raise TypeError(f"{array} is not one of ndarray, DataFrame and Series.")
        self._add_metric(name, array, loglevel)

    def add_any(self, name: str, obj: Any, loglevel: int | LogLevel = LogLevel.PERIODIC) -> None:
        """Log something with any type.

        As it's an "any" object, the only LogWriter accepting it is pickle.
        Therefore, pickle must be able to serialize it.
        """
        if loglevel < self._min_loglevel:
            return

        # FIXME: detect and rescue object that could be scalar or array

        self._add_metric(name, obj, loglevel)

    def logs(self) -> Dict[str, np.ndarray]:
        return {key: np.asanyarray(value, dtype="object") for key, value in self._logged.items()}


class LogWriter(Generic[ObsType, ActType]):
    """Base class for log writers, triggered at every reset and step by finite env.

    What to do with a specific log depends on the implementation of subclassing :class:`LogWriter`.
    The general principle is that, it should handle logs above its loglevel (inclusive),
    and discard logs that are not acceptable. For instance, console loggers obviously can't handle an image.
    """

    episode_count: int
    """Counter of episodes."""

    step_count: int
    """Counter of steps."""

    global_step: int
    """Counter of steps. Won"t be cleared in ``clear``."""

    global_episode: int
    """Counter of episodes. Won"t be cleared in ``clear``."""

    active_env_ids: Set[int]
    """Active environment ids in vector env."""

    episode_lengths: Dict[int, int]
    """Map from environment id to episode length."""

    episode_rewards: Dict[int, List[float]]
    """Map from environment id to episode total reward."""

    episode_logs: Dict[int, list]
    """Map from environment id to episode logs."""

    def __init__(self, loglevel: int | LogLevel = LogLevel.PERIODIC) -> None:
        self.loglevel = loglevel

        self.global_step = 0
        self.global_episode = 0

        # Information, logs of one episode is stored here.
        # This assumes that episode is not too long to fit into the memory.
        self.episode_lengths = dict()
        self.episode_rewards = dict()
        self.episode_logs = dict()

        self.clear()

    def clear(self):
        """Clear all the metrics for a fresh start.
        To make the logger instance reusable.
        """
        self.episode_count = self.step_count = 0
        self.active_env_ids = set()

    def state_dict(self) -> dict:
        """Save the states of the logger to a dict."""
        return {
            "episode_count": self.episode_count,
            "step_count": self.step_count,
            "global_step": self.global_step,
            "global_episode": self.global_episode,
            "active_env_ids": self.active_env_ids,
            "episode_lengths": self.episode_lengths,
            "episode_rewards": self.episode_rewards,
            "episode_logs": self.episode_logs,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load the states of current logger from a dict."""
        self.episode_count = state_dict["episode_count"]
        self.step_count = state_dict["step_count"]
        self.global_step = state_dict["global_step"]
        self.global_episode = state_dict["global_episode"]

        # These are runtime infos.
        # Though they are loaded, I don't think it really helps.
        self.active_env_ids = state_dict["active_env_ids"]
        self.episode_lengths = state_dict["episode_lengths"]
        self.episode_rewards = state_dict["episode_rewards"]
        self.episode_logs = state_dict["episode_logs"]

    @staticmethod
    def aggregation(array: Sequence[Any], name: str | None = None) -> Any:
        """Aggregation function from step-wise to episode-wise.

        If it's a sequence of float, take the mean.
        Otherwise, take the first element.

        If a name is specified and,

        - if it's ``reward``, the reduction will be sum.
        """
        assert len(array) > 0, "The aggregated array must be not empty."
        if all(isinstance(v, float) for v in array):
            if name == "reward":
                return np.sum(array)
            return np.mean(array)
        else:
            return array[0]

    def log_episode(self, length: int, rewards: List[float], contents: List[Dict[str, Any]]) -> None:
        """This is triggered at the end of each trajectory.

        Parameters
        ----------
        length
            Length of this trajectory.
        rewards
            A list of rewards at each step of this episode.
        contents
            Logged contents for every step.
        """

    def log_step(self, reward: float, contents: Dict[str, Any]) -> None:
        """This is triggered at each step.

        Parameters
        ----------
        reward
            Reward for this step.
        contents
            Logged contents for this step.
        """

    def on_env_step(self, env_id: int, obs: ObsType, rew: float, done: bool, info: InfoDict) -> None:
        """Callback for finite env, on each step."""

        # Update counter
        self.global_step += 1
        self.step_count += 1

        self.active_env_ids.add(env_id)
        self.episode_lengths[env_id] += 1
        # TODO: reward can be a list of list for MARL
        self.episode_rewards[env_id].append(rew)

        values: Dict[str, Any] = {}

        for key, (loglevel, value) in info["log"].items():
            if loglevel >= self.loglevel:  # FIXME: this is actually incorrect (see last FIXME)
                values[key] = value
        self.episode_logs[env_id].append(values)

        self.log_step(rew, values)

        if done:
            # Update counter
            self.global_episode += 1
            self.episode_count += 1

            self.log_episode(self.episode_lengths[env_id], self.episode_rewards[env_id], self.episode_logs[env_id])

    def on_env_reset(self, env_id: int, _: ObsType) -> None:
        """Callback for finite env.

        Reset episode statistics. Nothing task-specific is logged here because of
        `a limitation of tianshou <https://github.com/thu-ml/tianshou/issues/605>`__.
        """
        self.episode_lengths[env_id] = 0
        self.episode_rewards[env_id] = []
        self.episode_logs[env_id] = []

    def on_env_all_ready(self) -> None:
        """When all environments are ready to run.
        Usually, loggers should be reset here.
        """
        self.clear()

    def on_env_all_done(self) -> None:
        """All done. Time for cleanup."""


class LogBuffer(LogWriter):
    """Keep all numbers in memory.

    Objects that can't be aggregated like strings, tensors, images can't be stored in the buffer.
    To persist them, please use :class:`PickleWriter`.

    Every time, Log buffer receives a new metric, the callback is triggered,
    which is useful when tracking metrics inside a trainer.

    Parameters
    ----------
    callback
        A callback receiving three arguments:

        - on_episode: Whether it's called at the end of an episode
        - on_collect: Whether it's called at the end of a collect
        - log_buffer: the :class:`LogBbuffer` object

        No return value is expected.
    """

    # FIXME: needs a metric count

    def __init__(self, callback: Callable[[bool, bool, LogBuffer], None], loglevel: int | LogLevel = LogLevel.PERIODIC):
        super().__init__(loglevel)
        self.callback = callback

    def state_dict(self) -> dict:
        return {
            **super().state_dict(),
            "latest_metrics": self._latest_metrics,
            "aggregated_metrics": self._aggregated_metrics,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self._latest_metrics = state_dict["latest_metrics"]
        self._aggregated_metrics = state_dict["aggregated_metrics"]
        return super().load_state_dict(state_dict)

    def clear(self):
        super().clear()
        self._latest_metrics: dict[str, float] | None = None
        self._aggregated_metrics: dict[str, float] = defaultdict(float)

    def log_episode(self, length: int, rewards: list[float], contents: list[dict[str, Any]]) -> None:
        # FIXME Dup of ConsoleWriter
        episode_wise_contents: dict[str, list] = defaultdict(list)
        for step_contents in contents:
            for name, value in step_contents.items():
                # FIXME This could be false-negative for some numpy types
                if isinstance(value, float):
                    episode_wise_contents[name].append(value)

        logs: dict[str, float] = {}
        for name, values in episode_wise_contents.items():
            logs[name] = self.aggregation(values, name)  # type: ignore
            self._aggregated_metrics[name] += logs[name]

        self._latest_metrics = logs

        self.callback(True, False, self)

    def on_env_all_done(self) -> None:
        # This happens when collect exits
        self.callback(False, True, self)

    def episode_metrics(self) -> dict[str, float]:
        """Retrieve the numeric metrics of the latest episode."""
        if self._latest_metrics is None:
            raise ValueError("No episode metrics available yet.")
        return self._latest_metrics

    def collect_metrics(self) -> dict[str, float]:
        """Retrieve the aggregated metrics of the latest collect."""
        return {name: value / self.episode_count for name, value in self._aggregated_metrics.items()}


class ConsoleWriter(LogWriter):
    """Write log messages to console periodically.

    It tracks an average meter for each metric, which is the average value since last ``clear()`` till now.
    The display format for each metric is ``<name> <latest_value> (<average_value>)``.

    Non-single-number metrics are auto skipped.
    """

    prefix: str
    """Prefix can be set via ``writer.prefix``."""

    def __init__(
        self,
        log_every_n_episode: int = 20,
        total_episodes: int = None,
        float_format: str = ":.4f",
        counter_format: str = ":4d",
        loglevel: int | LogLevel = LogLevel.PERIODIC,
    ) -> None:
        super().__init__(loglevel)
        # TODO: support log_every_n_step
        self.log_every_n_episode = log_every_n_episode
        self.total_episodes = total_episodes

        self.counter_format = counter_format
        self.float_format = float_format

        self.prefix = ""

        self.console_logger = get_module_logger(__name__, level=logging.INFO)

    # FIXME: save & reload

    def clear(self) -> None:
        super().clear()
        # Clear average meters
        self.metric_counts: Dict[str, int] = defaultdict(int)
        self.metric_sums: Dict[str, float] = defaultdict(float)

    def log_episode(self, length: int, rewards: List[float], contents: List[Dict[str, Any]]) -> None:
        # Aggregate step-wise to episode-wise
        episode_wise_contents: Dict[str, list] = defaultdict(list)

        for step_contents in contents:
            for name, value in step_contents.items():
                if isinstance(value, float):
                    episode_wise_contents[name].append(value)

        # Generate log contents and track them in average-meter.
        # This should be done at every step, regardless of periodic or not.
        logs: Dict[str, float] = {}
        for name, values in episode_wise_contents.items():
            logs[name] = self.aggregation(values, name)  # type: ignore

        for name, value in logs.items():
            self.metric_counts[name] += 1
            self.metric_sums[name] += value

        if self.episode_count % self.log_every_n_episode == 0 or self.episode_count == self.total_episodes:
            # Only log periodically or at the end
            self.console_logger.info(self.generate_log_message(logs))

    def generate_log_message(self, logs: Dict[str, float]) -> str:
        if self.prefix:
            msg_prefix = self.prefix + " "
        else:
            msg_prefix = ""
        if self.total_episodes is None:
            msg_prefix += "[Step {" + self.counter_format + "}]"
        else:
            msg_prefix += "[{" + self.counter_format + "}/" + str(self.total_episodes) + "]"
        msg_prefix = msg_prefix.format(self.episode_count)

        msg = ""
        for name, value in logs.items():
            # Double-space as delimiter
            format_template = r"  {} {" + self.float_format + "} ({" + self.float_format + "})"
            msg += format_template.format(name, value, self.metric_sums[name] / self.metric_counts[name])

        msg = msg_prefix + " " + msg

        return msg


class CsvWriter(LogWriter):
    """Dump all episode metrics to a ``result.csv``.

    This is not the correct implementation. It's only used for first iteration.
    """

    SUPPORTED_TYPES = (float, str, pd.Timestamp)

    all_records: List[Dict[str, Any]]

    # FIXME: save & reload

    def __init__(self, output_dir: Path, loglevel: int | LogLevel = LogLevel.PERIODIC) -> None:
        super().__init__(loglevel)
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)

    def clear(self) -> None:
        super().clear()
        self.all_records = []

    def log_episode(self, length: int, rewards: List[float], contents: List[Dict[str, Any]]) -> None:
        # FIXME Same as ConsoleLogger, needs a refactor to eliminate code-dup
        episode_wise_contents: Dict[str, list] = defaultdict(list)

        for step_contents in contents:
            for name, value in step_contents.items():
                if isinstance(value, self.SUPPORTED_TYPES):
                    episode_wise_contents[name].append(value)

        logs: Dict[str, float] = {}
        for name, values in episode_wise_contents.items():
            logs[name] = self.aggregation(values, name)  # type: ignore

        self.all_records.append(logs)

    def on_env_all_done(self) -> None:
        # FIXME: this is temporary
        pd.DataFrame.from_records(self.all_records).to_csv(self.output_dir / "result.csv", index=False)


# The following are not implemented yet.


class PickleWriter(LogWriter):
    """Dump logs to pickle files."""


class TensorboardWriter(LogWriter):
    """Write logs to event files that can be visualized with tensorboard."""


class MlflowWriter(LogWriter):
    """Add logs to mlflow."""
