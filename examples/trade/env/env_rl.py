import gym

gym.logger.set_level(40)
import numpy as np
import pandas as pd
import pickle as pkl
import datetime
import random
import os
import json
import time
import tianshou as ts
import copy
from multiprocessing import Process, Pipe, Queue
from typing import List, Tuple, Union, Optional, Callable, Any
from tianshou.env.utils import CloudpickleWrapper
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

import sys

sys.path.append("..")
from util import merge_dicts, nan_weighted_avg, robust_auc
import reward
import observation
import action

ZERO = 1e-7


class StockEnv(gym.Env):
    """Single-assert environment"""

    def __init__(self, config):
        self.max_step_num = config["max_step_num"]
        self.limit = config["limit"]
        self.time_interval = config["time_interval"]
        self.interval_num = config["interval_num"]
        self.offset = config["offset"] if "offset" in config else 0
        if "last_reward" in config:
            self.last_reward = config["last_reward"]
        else:
            self.last_reward = None
        if "log" in config:
            self.log = config["log"]
        else:
            self.log = True
        # loader_conf = config['loader']['config']
        obs_conf = config["obs"]["config"]
        obs_conf["features"] = config["features"]
        obs_conf["time_interval"] = self.time_interval
        obs_conf["max_step_num"] = self.max_step_num
        self.obs = getattr(observation, config["obs"]["name"])(obs_conf)
        self.action_func = getattr(action, config["action"]["name"])(config["action"]["config"])
        self.reward_func_list = []
        self.reward_log_dict = {}
        self.reward_coef = []
        for name, conf in config["reward"].items():
            self.reward_coef.append(conf.pop("coefficient"))
            self.reward_func_list.append(getattr(reward, name)(conf))
            self.reward_log_dict[name] = 0.0
        self.observation_space = self.obs.get_space()
        self.action_space = self.action_func.get_space()

    def toggle_log(self, log):
        self.log = log

    def reset(self, sample):
        """

        :param sample:

        """

        for key in self.reward_log_dict.keys():
            self.reward_log_dict[key] = 0.0
        if not sample is None:
            (
                self.ins,
                self.date,
                self.raw_df_values,
                self.raw_df_columns,
                self.raw_df_index,
                self.feature_dfs,
                self.target,
                self.is_buy,
            ) = sample
        self.raw_df = pd.DataFrame(index=self.raw_df_index, data=self.raw_df_values, columns=self.raw_df_columns,)
        del self.raw_df_values, self.raw_df_columns, self.raw_df_index
        start_time = time.time()
        self.load_time = time.time() - start_time
        self.day_vwap = nan_weighted_avg(
            self.raw_df["$vwap0"].values[self.offset : self.offset + self.max_step_num],
            self.raw_df["$volume0"].values[self.offset : self.offset + self.max_step_num],
        )
        try:
            assert not (np.isnan(self.day_vwap) or np.isinf(self.day_vwap))
        except:
            print(self.raw_df)
            print(self.ins)
            print(self.day_vwap)
            self.raw_df.to_pickle("/nfs_data1/kanren/error_df.pkl")
        self.day_twap = np.nanmean(self.raw_df["$vwap0"].values[self.offset : self.offset + self.max_step_num])
        self.t = -1 + self.offset
        self.interval = 0
        self.position = self.target
        self.eps_start = time.time()

        self.state = self.obs(
            self.raw_df,
            self.feature_dfs,
            self.t,
            self.interval,
            self.position,
            self.target,
            self.is_buy,
            self.max_step_num,
            self.interval_num,
        )
        if self.log:
            index_array = [
                np.array([self.ins] * self.max_step_num),
                self.raw_df.index.to_numpy()[self.offset : self.offset + self.max_step_num],
                np.array([self.date] * self.max_step_num),
            ]
            self.traded_log = pd.DataFrame(
                data={
                    "$v_t": np.nan,
                    "$max_vol_t": (self.raw_df["$volume0"] * self.limit).values[
                        self.offset : self.offset + self.max_step_num
                    ],
                    "$traded_t": np.nan,
                    "$vwap_t": self.raw_df["$vwap0"].values[self.offset : self.offset + self.max_step_num],
                    "action": np.nan,
                },
                index=index_array,
            )
        # v_t: The amount of shares the agent hope to trade
        # max_vol_t: The max amount of shares can be traded
        # traded_t: The amount of shares that is acually traded
        # action: the action of agent, may have various meanings in different settings.
        self.done = False
        if self.limit > 1:
            self.this_valid = np.inf
        else:
            self.this_valid = np.nansum(self.raw_df["$volume0"].values) * self.limit
        self.this_cash = 0

        self.step_time = []
        self.action_log = [np.nan] * self.interval_num
        self.reset_time = time.time() - start_time
        self.real_eps_time = self.reset_time
        self.total_reward = 0
        self.total_instant_rew = 0
        self.last_rew = 0
        return self.state

    def step(self, action):
        """

        :param action:

        """
        start_time = time.time()
        self.action_log[self.interval] = action
        volume_t = self.action_func(
            action,
            self.target,
            self.position,
            max_step_num=self.max_step_num,
            t=self.t - self.offset,
            interval=self.interval,
            interval_num=self.interval_num,
        )
        self.interval += 1
        reward = 0.0
        time_left = self.max_step_num - self.t - 1 + self.offset

        for i in range(self.time_interval):
            v_t = volume_t / min(self.time_interval, time_left)
            self.t += 1
            if self.t == self.max_step_num - 1 + self.offset:
                v_t = self.position
            if self.log:
                log_index = self.t - self.offset
                self.traded_log.iat[log_index, 0] = v_t
                self.traded_log.iat[log_index, 4] = action
            vwap_t, vol_t = self.raw_df.iloc[self.t][["$vwap0", "$volume0"]]
            max_vol_t = self.limit * vol_t
            if self.limit >= 1:
                max_vol_t = np.inf
            if v_t > min(self.position, max_vol_t):
                if self.position <= max_vol_t:
                    v_t = self.position
                else:
                    v_t = max_vol_t
            self.position -= v_t
            self.this_cash += vwap_t * v_t
            if self.log:
                self.traded_log.iat[log_index, 2] = v_t

            if self.is_buy:
                performance_raise = (1 - vwap_t / self.day_vwap) * 10000
                PA_t = (1 - vwap_t / self.day_twap) * 10000
            else:
                performance_raise = (vwap_t / self.day_vwap - 1) * 10000
                PA_t = (vwap_t / self.day_twap - 1) * 10000

            for i, reward_func in enumerate(self.reward_func_list):
                if reward_func.isinstant:
                    tmp_r = reward_func(performance_raise, v_t, self.target, PA_t)
                    reward += tmp_r * self.reward_coef[i]
                    self.reward_log_dict[type(reward_func).__name__] += tmp_r

            if self.t == self.max_step_num - 1 + self.offset:
                break

        if self.position < ZERO:
            self.done = True

        if self.interval == self.interval_num:
            self.done = True

        self.step_time.append(time.time() - start_time)
        self.real_eps_time += time.time() - start_time
        if self.done:
            this_traded = self.target - self.position
            this_vwap = (self.this_cash / this_traded) if this_traded > ZERO else self.day_vwap
            valid = min(self.target, self.this_valid)
            this_ffr = (this_traded / valid) if valid > ZERO else 1.0
            if abs(this_ffr - 1.0) < ZERO:
                this_ffr = 1.0
            this_ffr *= 100
            this_vv_ratio = this_vwap / self.day_vwap
            vwap = self.raw_df["$vwap0"].values[self.offset : self.max_step_num + self.offset]
            this_tt_ratio = this_vwap / np.nanmean(vwap)

            if self.is_buy:
                performance_raise = (1 - this_vv_ratio) * 10000
                PA = (1 - this_tt_ratio) * 10000
            else:
                performance_raise = (this_vv_ratio - 1) * 10000
                PA = (this_tt_ratio - 1) * 10000

            for i, reward_func in enumerate(self.reward_func_list):
                if not reward_func.isinstant:
                    tmp_r = reward_func(performance_raise, this_ffr, this_tt_ratio, self.is_buy)
                    reward += tmp_r * self.reward_coef[i]
                    self.reward_log_dict[type(reward_func).__name__] += tmp_r

            self.state = self.obs(
                self.raw_df,
                self.feature_dfs,
                self.t,
                self.interval,
                self.position,
                self.target,
                self.is_buy,
                self.max_step_num,
                self.interval_num,
                action,
            )
            if self.log:
                res = pd.DataFrame(
                    {
                        "target": self.target,
                        "sell": not self.is_buy,
                        "vwap": this_vwap,
                        "this_vv_ratio": this_vv_ratio,
                        "this_ffr": this_ffr,
                    },
                    index=[[self.ins], [self.date]],
                )
            money = self.target * self.day_vwap
            if self.is_buy:
                info = {
                    "money": money,
                    "money_buy": money,
                    "action": self.action_log,
                    "ffr": this_ffr,
                    "obs0_PR": performance_raise,
                    "ffr_buy": this_ffr,
                    "PR_buy": performance_raise,
                    "PA": PA,
                    "PA_buy": PA,
                    "vwap": this_vwap,
                }
            else:
                info = {
                    "money": money,
                    "money_sell": money,
                    "action": self.action_log,
                    "ffr": this_ffr,
                    "obs0_PR": performance_raise,
                    "ffr_sell": this_ffr,
                    "PR_sell": performance_raise,
                    "PA": PA,
                    "PA_sell": PA,
                    "vwap": this_vwap,
                }
            info = merge_dicts(info, self.reward_log_dict)
            if self.log:
                info["df"] = self.traded_log
                info["res"] = res
            del self.feature_dfs
            return self.state, reward, self.done, info

        else:
            self.state = self.obs(
                self.raw_df,
                self.feature_dfs,
                self.t,
                self.interval,
                self.position,
                self.target,
                self.is_buy,
                self.max_step_num,
                self.interval_num,
                action,
            )
            return self.state, reward, self.done, {}


class StockEnv_Acc(StockEnv):
    def step(self, action):
        start_time = time.time()
        self.action_log[self.interval] = action
        volume_t = self.action_func(
            action,
            self.target,
            self.position,
            max_step_num=self.max_step_num,
            t=self.t - self.offset,
            interval=self.interval,
            interval_num=self.interval_num,
        )
        self.interval += 1
        reward = 0.0
        time_left = self.max_step_num - self.t - 1 + self.offset
        time_left = min(self.time_interval, time_left)

        v_t = np.repeat(volume_t / time_left, time_left)
        minutes = np.arange(self.t + 1, self.t + time_left + 1)
        if self.log:
            log_index = minutes - self.offset
            self.traded_log.iloc[log_index, 0] = v_t
            self.traded_log.iloc[log_index, 4] = action
        vwap_t = self.raw_df.iloc[minutes]["$vwap0"].values
        vol_t = self.raw_df.iloc[minutes]["$volume0"].values
        max_vol_t = self.limit * vol_t if self.limit < 1 else np.inf
        v_t = np.minimum(v_t, max_vol_t)
        if self.t + time_left == self.max_step_num - 1 + self.offset:
            left = self.position - v_t.sum()
            v_t[-1] += left
            v_t = np.minimum(v_t, max_vol_t)
        this_money = (v_t * vwap_t).sum()
        this_vol = v_t.sum()
        this_vwap = np.nan_to_num(this_money / this_vol)
        self.t += time_left
        self.position -= this_vol
        self.this_cash += this_money
        if self.log:
            self.traded_log.iloc[log_index, 2] = v_t

        if self.is_buy:
            performance_raise = (1 - this_vwap / self.day_vwap) * 10000
            PA_t = (1 - this_vwap / self.day_twap) * 10000
        else:
            performance_raise = (this_vwap / self.day_vwap - 1) * 10000
            PA_t = (this_vwap / self.day_twap - 1) * 10000

        for i, reward_func in enumerate(self.reward_func_list):
            if reward_func.isinstant:
                tmp_r = reward_func(performance_raise, v_t, self.target, PA_t)
                reward += tmp_r * self.reward_coef[i]
                self.reward_log_dict[type(reward_func).__name__] += tmp_r

        if self.position < ZERO:
            self.done = True

        if self.interval == self.interval_num:
            self.done = True

        self.step_time.append(time.time() - start_time)
        self.real_eps_time += time.time() - start_time
        if self.done:
            this_traded = self.target - self.position
            this_vwap = (self.this_cash / this_traded) if this_traded > ZERO else self.day_vwap
            valid = min(self.target, self.this_valid)
            this_ffr = (this_traded / valid) if valid > ZERO else 1.0
            if abs(this_ffr - 1.0) < ZERO:
                this_ffr = 1.0
            this_ffr *= 100
            this_vv_ratio = this_vwap / self.day_vwap
            vwap = self.raw_df["$vwap0"].values[self.offset : self.max_step_num + self.offset]
            this_tt_ratio = this_vwap / np.nanmean(vwap)

            if self.is_buy:
                performance_raise = (1 - this_vv_ratio) * 10000
                PA = (1 - this_tt_ratio) * 10000
            else:
                performance_raise = (this_vv_ratio - 1) * 10000
                PA = (this_tt_ratio - 1) * 10000

            for i, reward_func in enumerate(self.reward_func_list):
                if not reward_func.isinstant:
                    tmp_r = reward_func(performance_raise, this_ffr, this_tt_ratio, self.is_buy)
                    reward += tmp_r * self.reward_coef[i]
                    self.reward_log_dict[type(reward_func).__name__] += tmp_r

            self.state = self.obs(
                self.raw_df,
                self.feature_dfs,
                self.t,
                self.interval,
                self.position,
                self.target,
                self.is_buy,
                self.max_step_num,
                self.interval_num,
                action,
            )
            if self.log:
                res = pd.DataFrame(
                    {
                        "target": self.target,
                        "sell": not self.is_buy,
                        "vwap": this_vwap,
                        "this_vv_ratio": this_vv_ratio,
                        "this_ffr": this_ffr,
                    },
                    index=[[self.ins], [self.date]],
                )
            money = self.target * self.day_vwap
            if self.is_buy:
                info = {
                    "money": money,
                    "money_buy": money,
                    "action": self.action_log,
                    "ffr": this_ffr,
                    "obs0_PR": performance_raise,
                    "ffr_buy": this_ffr,
                    "PR_buy": performance_raise,
                    "PA": PA,
                    "PA_buy": PA,
                    "vwap": this_vwap,
                }
            else:
                info = {
                    "money": money,
                    "money_sell": money,
                    "action": self.action_log,
                    "ffr": this_ffr,
                    "obs0_PR": performance_raise,
                    "ffr_sell": this_ffr,
                    "PR_sell": performance_raise,
                    "PA": PA,
                    "PA_sell": PA,
                    "vwap": this_vwap,
                }
            info = merge_dicts(info, self.reward_log_dict)
            if self.log:
                info["df"] = self.traded_log
                info["res"] = res
            del self.feature_dfs
            return self.state, reward, self.done, info

        else:
            self.state = self.obs(
                self.raw_df,
                self.feature_dfs,
                self.t,
                self.interval,
                self.position,
                self.target,
                self.is_buy,
                self.max_step_num,
                self.interval_num,
                action,
            )
            return self.state, reward, self.done, {}
