# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Log RL env steps: reset and step."""

from qlib.log import get_module_logger


class RLLogger:
    def log_step(self, env_id, obs, rew, done, info):
        raise NotImplementedError()

    def log_reset(self, env_id, obs):
        raise NotImplementedError()


import inspect
import json
from collections import defaultdict
from typing import TextIO

import numpy as np
import pandas as pd
from torch.utils.tensorboard.writer import SummaryWriter

from .finite_env import BaseLogger


__all__ = ["Logger"]

_tb_logger = _json_writer = None


def _get_tb_logger() -> SummaryWriter:
    global _tb_logger
    if _tb_logger is None:
        _tb_logger = SummaryWriter(log_dir=get_tb_log_dir())
    return _tb_logger


def _get_json_writer() -> TextIO:
    global _json_writer
    if _json_writer is None:
        _json_writer = (get_output_dir() / "summary.json").open("a")
    return _json_writer


def _groupby_category(category, value, key) -> defaultdict:
    """
    Group the values by category.
    """
    if not isinstance(value, (list, tuple, np.ndarray)):
        value = [value] * len(category)
    assert len(category) == len(value)
    grouped = defaultdict(list)
    for c, v in zip(category, value):
        grouped[f"{key}/{c}"].append(v)
    return grouped


class Logger(BaseLogger):
    def __init__(
        self, ep_total, *, log_interval=100, prefix="Episode", tb_prefix="", count_global="episode", reward_func=np.mean
    ):
        self.meter = MetricMeter()
        self.ep_count = 0
        self.global_step = 0
        self.ep_total = ep_total
        self.log_interval = log_interval
        self.prefix = prefix
        self.logs = []
        self.history = []
        self.active_env_ids = set()
        assert count_global in ["step", "episode"]
        self.count_global = count_global

        self.tb_writer = _get_tb_logger()
        self.tb_prefix = tb_prefix

        self.json_writer = _get_json_writer()

        self.episode_lengths = dict()
        self.episode_rewards = dict()
        self.episode_rewards_info = dict()
        self.reward_func = reward_func

    def log_step(self, env_id, obs, rew, done, info):
        self.active_env_ids.add(env_id)
        self.episode_lengths[env_id] += 1
        self.episode_rewards[env_id] += self.reward_func(rew)

        for k, v in info.get("reward", {}).items():
            self.episode_rewards_info[env_id][k] += v

        if self.count_global == "step":
            self.global_step += 1

        if not done:
            return

        if self.count_global == "episode":
            self.global_step += 1

        self.ep_count += 1
        index = dict(info["index"])
        logs = dict(info["logs"])  # deal with batch
        logs.update(
            {
                "step_per_episode": self.episode_lengths[env_id],
                "reward": self.episode_rewards[env_id],
                "num_active_envs": len(self.active_env_ids),
            }
        )
        logs.update({f"reward/{k}": v for k, v in self.episode_rewards_info[env_id].items()})

        # TODO: meter.update support array input

        category = info.get("category", 'default')
        if not isinstance(category, (list, tuple, np.ndarray)):
            category = [category]
        cate_logs = {}
        for k, v in logs.items():
            cate_logs.update(_groupby_category(category, v, k))
        self.meter.update({k: np.nanmean(v) for k, v in cate_logs.items()})

        self.meter.update({k: np.nanmean(v) for k, v in logs.items()})
        self.logs += pd.DataFrame({**index, **logs, "category": category}).to_dict(orient="records")
        self.history.append({**index, **info["history"]})
        if self.ep_count % self.log_interval == 0 or self.ep_count >= self.ep_total:
            frm = inspect.stack()[1]
            mod = inspect.getmodule(frm[0])
            print_log(f"{self.prefix} [{self.ep_count}/{self.ep_total}]  {self.meter}", mod.__name__)

    def log_reset(self, env_id, obs):
        self.episode_lengths[env_id] = 0
        self.episode_rewards[env_id] = 0.0
        self.episode_rewards_info[env_id] = defaultdict(float)

    def set_prefix(self, prefix):
        self.prefix = prefix

    def write_summary(self, extra_metrics=None):
        if extra_metrics:
            self.meter.update(extra_metrics)
        summary = self.summary()
        print_log(f"{self.prefix} Summary:\n" + "\n".join([f"    {k}\t{v:.4f}" for k, v in summary.items()]), __name__)
        for key, value in summary.items():
            if self.tb_prefix:
                key = self.tb_prefix + "/" + key
            self.tb_writer.add_scalar(key, value, global_step=self.global_step)
        summary = {"prefix": self.tb_prefix, "step": self.global_step, **summary}
        self.json_writer.write(json.dumps(summary) + "\n")
        self.json_writer.flush()

    def summary(self):
        return {key: self.meter[key].avg for key in self.meter}

    def reset(self, prefix=None):
        self.ep_count = self.step_count = 0
        self.meter.reset()
        self.active_env_ids = set()
        self.logs = []
        if prefix is not None:
            self.set_prefix(prefix)

    def state_dict(self):
        # logging status within epoch is not saved
        return {
            "global_step": self.global_step,
        }

    def load_state_dict(self, state_dict):
        self.global_step = state_dict["global_step"]