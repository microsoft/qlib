# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import argparse
import os
import random
import sys
import warnings
from pathlib import Path
from typing import Any, cast, List, Optional

import numpy as np
import pandas as pd
import torch
from qlib.backtest import Order
from qlib.backtest.decision import OrderDir
from qlib.constant import ONE_MIN
from qlib.rl.contrib.naive_config_parser import TrainingConfigParser
from qlib.rl.data.integration import init_qlib
from qlib.rl.data.pickle_styled import load_pickle_intraday_processed_data
from qlib.rl.interpreter import ActionInterpreter, StateInterpreter
from qlib.rl.order_execution import SingleAssetOrderExecutionSimple
from qlib.rl.order_execution.simulator_qlib import SingleAssetOrderExecution
from qlib.rl.reward import Reward
from qlib.rl.trainer import Checkpoint, backtest, train
from qlib.rl.trainer.callbacks import Callback, EarlyStopping, MetricsWriter
from qlib.rl.utils.log import CsvWriter
from qlib.utils import init_instance_by_config
from tianshou.policy import BasePolicy
from torch.utils.data import Dataset


def get_executor_config(freq: int) -> dict:
    return {
        "class": "NestedExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {
            "inner_executor": {
                "class": "NestedExecutor",
                "module_path": "qlib.backtest.executor",
                "kwargs": {
                    "inner_executor": {
                        "class": "SimulatorExecutor",
                        "module_path": "qlib.backtest.executor",
                        "kwargs": {
                            "generate_report": False,
                            "time_per_step": f"{freq}min",
                            "track_data": True,
                            "trade_type": "serial",
                            "verbose": False,
                        },
                    },
                    "inner_strategy": {
                        "class": "TWAPStrategy",
                        "kwargs": {},
                        "module_path": "qlib.contrib.strategy.rule_strategy",
                    },
                    "time_per_step": "30min",
                    "track_data": True,
                },
            },
            "inner_strategy": {
                "class": "ProxySAOEStrategy",
                "module_path": "qlib.rl.order_execution.strategy",
                "kwargs": {},
            },
            "time_per_step": "1day",
            "track_data": True,
        },
    }


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def _read_orders(order_dir: Path) -> pd.DataFrame:
    if os.path.isfile(order_dir):
        return pd.read_pickle(order_dir)
    else:
        orders = []
        for file in order_dir.iterdir():
            order_data = pd.read_pickle(file)
            orders.append(order_data)
        return pd.concat(orders)


def _freq_str_to_int(freq: str) -> int:
    if freq.endswith("min"):
        return int(freq.replace("min", ""))
    elif freq.endswith("hour"):
        return int(freq.replace("hour", "") * 60)
    else:
        raise ValueError(f"Unrecognized freq string: {freq}")


class LazyLoadDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        order_df: pd.DataFrame,
        default_start_time_index: int,
        default_end_time_index: int,
    ) -> None:
        self._default_start_time_index = default_start_time_index
        self._default_end_time_index = default_end_time_index

        self._order_df = order_df
        self._ticks_index: Optional[pd.DatetimeIndex] = None
        self._data_dir = Path(data_dir)

    def __len__(self) -> int:
        return len(self._order_df)

    def __getitem__(self, index: int) -> Order:
        row = self._order_df.iloc[index]
        date = pd.Timestamp(str(row["date"]))

        if self._ticks_index is None:
            # TODO: We only load ticks index once based on the assumption that ticks index of different dates
            # TODO: in one experiment are all the same. If that assumption is not hold, we need to load ticks index
            # TODO: of all dates.

            data = load_pickle_intraday_processed_data(
                data_dir=self._data_dir,
                stock_id=row["instrument"],
                date=date,
                feature_columns_today=[],
                feature_columns_yesterday=[],
                backtest=True,
            )
            self._ticks_index = [t - date for t in data.today.index]

        order = Order(
            stock_id=row["instrument"],
            amount=row["amount"],
            direction=OrderDir(int(row["order_type"])),
            start_time=date + self._ticks_index[self._default_start_time_index],
            end_time=date + self._ticks_index[self._default_end_time_index - 1] + ONE_MIN,
        )

        return order


def _split_order_df_by_instrument(df: pd.DataFrame, k: int) -> List[pd.DataFrame]:
    df = df.copy()
    df["group"] = df["instrument"].apply(lambda s: hash(s) % k)
    dfs = [df[df['group'] == i].drop(columns=["group"]) for i in range(k)]
    return dfs


def train_and_test(
    freq: str,
    concurrency: int,
    parallel_mode: str,
    training_config: dict,
    simulator_config: dict,
    policy: BasePolicy,
    state_interpreter: StateInterpreter,
    action_interpreter: ActionInterpreter,
    reward: Reward,
    run_training: bool,
    run_backtest: bool,
) -> None:
    freq = _freq_str_to_int(freq)
    order_root_path = Path(training_config["order_dir"])
    feature_root_dir = simulator_config["data"]["feature_root_dir"]
    assert simulator_config["data"]["default_start_time_index"] % freq == 0
    assert simulator_config["data"]["default_end_time_index"] % freq == 0

    sim_type = simulator_config["type"]
    if sim_type == "simple":

        def _simulator_factory(order: Order) -> SingleAssetOrderExecutionSimple:
            simulator = SingleAssetOrderExecutionSimple(
                order=order,
                data_dir=feature_root_dir,
                feature_columns_today=simulator_config["data"]["feature_columns_today"],
                data_granularity=freq,
                ticks_per_step=simulator_config["time_per_step"],
                vol_threshold=simulator_config["vol_limit"],
            )
            return simulator

    elif sim_type == "full":
        init_qlib(simulator_config["qlib"])
        executor_config = get_executor_config(freq)
        exchange_config = simulator_config["exchange"]

        def _simulator_factory(order: Order) -> SingleAssetOrderExecution:
            simulator = SingleAssetOrderExecution(
                order=order,
                executor_config=executor_config,
                exchange_config={**exchange_config, **{"codes": [order.stock_id]}},
                qlib_config=None,
                cash_limit=None,
            )
            return simulator

    # Load orders
    load_data_tags = []
    orders_by_tag = {}
    if run_training:
        load_data_tags += ["train", "valid"]
    if run_backtest:
        load_data_tags += ["test"]
    for tag in load_data_tags:
        order_df = _read_orders(order_root_path / tag).reset_index()
        dfs = _split_order_df_by_instrument(order_df, concurrency)
        datasets = [
            LazyLoadDataset(
                data_dir=feature_root_dir,
                order_df=df,
                default_start_time_index=simulator_config["data"]["default_start_time_index"] // freq,
                default_end_time_index=simulator_config["data"]["default_end_time_index"] // freq,
            )
            for df in dfs
        ]
        orders_by_tag[tag] = datasets

    if run_training:
        callbacks: List[Callback] = [
            MetricsWriter(dirpath=Path(training_config["checkpoint_path"])),
            Checkpoint(
                dirpath=Path(training_config["checkpoint_path"]) / "checkpoints",
                every_n_iters=training_config["checkpoint_every_n_iters"],
                save_latest="copy",
            ),
            EarlyStopping(
                patience=training_config["earlystop_patience"],
                monitor="val/pa",
            ),
        ]

        train(
            simulator_fn=_simulator_factory,
            state_interpreter=state_interpreter,
            action_interpreter=action_interpreter,
            policy=policy,
            reward=reward,
            initial_states=cast(List[List[Order]], orders_by_tag["train"]),
            trainer_kwargs={
                "max_iters": training_config["max_epoch"],
                "finite_env_type": parallel_mode,
                "concurrency": concurrency,
                "val_every_n_iters": training_config["val_every_n_epoch"],
                "callbacks": callbacks,
            },
            vessel_kwargs={
                "episode_per_iter": training_config["episode_per_collect"],
                "update_kwargs": {
                    "batch_size": training_config["batch_size"],
                    "repeat": training_config["repeat_per_collect"],
                },
                "val_initial_states": cast(List[List[Order]], orders_by_tag["valid"]),
            },
        )

    if run_backtest:
        backtest(
            simulator_fn=_simulator_factory,
            state_interpreter=state_interpreter,
            action_interpreter=action_interpreter,
            initial_states=cast(List[List[Order]], orders_by_tag["test"]),
            policy=policy,
            logger=CsvWriter(Path(training_config["checkpoint_path"])),
            reward=reward,
            finite_env_type=parallel_mode,
            concurrency=concurrency,
        )


def main(config: dict, run_training: bool, run_backtest: bool) -> None:
    if not run_training and not run_backtest:
        warnings.warn("Skip the entire job since training and backtest are both skipped.")
        return

    seed = config["runtime"]["seed"]
    if seed is not None:
        seed_everything(seed)

    for extra_module_path in config["general"]["extra_module_paths"]:
        sys.path.append(extra_module_path)

    state_interpreter: StateInterpreter = init_instance_by_config(config["interpreter"]["state"])
    action_interpreter: ActionInterpreter = init_instance_by_config(config["interpreter"]["action"])
    reward: Reward = init_instance_by_config(config["interpreter"]["reward"])

    additional_policy_kwargs = {
        "obs_space": state_interpreter.observation_space,
        "action_space": action_interpreter.action_space,
    }
    # Create torch network
    if "network" in config["policy"]:
        network_config = config["policy"]["network"]
        network_config["kwargs"] = {
            **network_config.get("kwargs", {}),
            **{"obs_space": state_interpreter.observation_space},
        }
        additional_policy_kwargs["network"] = init_instance_by_config(network_config)

    # Create policy
    policy_config = config["policy"]["policy"]
    policy_config["kwargs"] = {**policy_config.get("kwargs", {}), **additional_policy_kwargs}
    policy: BasePolicy = init_instance_by_config(policy_config)

    use_cuda = config["runtime"]["use_cuda"]
    if use_cuda:
        policy.cuda()

    train_and_test(
        freq=config["general"]["freq"],
        concurrency=config["runtime"]["concurrency"],
        parallel_mode=config["runtime"]["parallel_mode"],
        training_config=config["training"],
        simulator_config=config["simulator"],
        policy=policy,
        state_interpreter=state_interpreter,
        action_interpreter=action_interpreter,
        reward=reward,
        run_training=run_training,
        run_backtest=run_backtest,
    )


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True, help="Path to the config file")
    parser.add_argument("--no_training", action="store_true", help="Skip training workflow.")
    parser.add_argument("--run_backtest", action="store_true", help="Run backtest workflow.")
    args = parser.parse_args()

    config_parser = TrainingConfigParser(args.config_path)
    config = config_parser.parse()
    main(config, run_training=not args.no_training, run_backtest=args.run_backtest)
