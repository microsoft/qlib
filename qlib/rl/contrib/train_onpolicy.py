# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import argparse
import os
import random
import warnings
from pathlib import Path
from typing import cast, List, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from qlib.backtest import Order
from qlib.backtest.decision import OrderDir
from qlib.constant import ONE_MIN
from qlib.rl.data.native import load_handler_intraday_processed_data
from qlib.rl.interpreter import ActionInterpreter, StateInterpreter
from qlib.rl.order_execution import SingleAssetOrderExecutionSimple
from qlib.rl.reward import Reward
from qlib.rl.trainer import Checkpoint, backtest, train
from qlib.rl.trainer.callbacks import Callback, EarlyStopping, MetricsWriter
from qlib.rl.utils.log import CsvWriter
from qlib.utils import init_instance_by_config
from tianshou.policy import BasePolicy
from torch.utils.data import Dataset


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


class LazyLoadDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        order_file_path: Path,
        default_start_time_index: int,
        default_end_time_index: int,
    ) -> None:
        self._default_start_time_index = default_start_time_index
        self._default_end_time_index = default_end_time_index

        self._order_df = _read_orders(order_file_path).reset_index()
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

            data = load_handler_intraday_processed_data(
                data_dir=self._data_dir,
                stock_id=row["instrument"],
                date=date,
                feature_columns_today=[],
                feature_columns_yesterday=[],
                backtest=True,
                index_only=True,
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


def train_and_test(
    env_config: dict,
    simulator_config: dict,
    trainer_config: dict,
    data_config: dict,
    state_interpreter: StateInterpreter,
    action_interpreter: ActionInterpreter,
    policy: BasePolicy,
    reward: Reward,
    run_training: bool,
    run_backtest: bool,
) -> None:
    order_root_path = Path(data_config["source"]["order_dir"])

    data_granularity = simulator_config.get("data_granularity", 1)

    def _simulator_factory_simple(order: Order) -> SingleAssetOrderExecutionSimple:
        return SingleAssetOrderExecutionSimple(
            order=order,
            data_dir=data_config["source"]["feature_root_dir"],
            feature_columns_today=data_config["source"]["feature_columns_today"],
            feature_columns_yesterday=data_config["source"]["feature_columns_yesterday"],
            data_granularity=data_granularity,
            ticks_per_step=simulator_config["time_per_step"],
            vol_threshold=simulator_config["vol_limit"],
        )

    assert data_config["source"]["default_start_time_index"] % data_granularity == 0
    assert data_config["source"]["default_end_time_index"] % data_granularity == 0

    if run_training:
        train_dataset, valid_dataset = [
            LazyLoadDataset(
                data_dir=data_config["source"]["feature_root_dir"],
                order_file_path=order_root_path / tag,
                default_start_time_index=data_config["source"]["default_start_time_index"] // data_granularity,
                default_end_time_index=data_config["source"]["default_end_time_index"] // data_granularity,
            )
            for tag in ("train", "valid")
        ]

        callbacks: List[Callback] = []
        if "checkpoint_path" in trainer_config:
            callbacks.append(MetricsWriter(dirpath=Path(trainer_config["checkpoint_path"])))
            callbacks.append(
                Checkpoint(
                    dirpath=Path(trainer_config["checkpoint_path"]) / "checkpoints",
                    every_n_iters=trainer_config.get("checkpoint_every_n_iters", 1),
                    save_latest="copy",
                ),
            )
        if "earlystop_patience" in trainer_config:
            callbacks.append(
                EarlyStopping(
                    patience=trainer_config["earlystop_patience"],
                    monitor="val/pa",
                )
            )

        train(
            simulator_fn=_simulator_factory_simple,
            state_interpreter=state_interpreter,
            action_interpreter=action_interpreter,
            policy=policy,
            reward=reward,
            initial_states=cast(List[Order], train_dataset),
            trainer_kwargs={
                "max_iters": trainer_config["max_epoch"],
                "finite_env_type": env_config["parallel_mode"],
                "concurrency": env_config["concurrency"],
                "val_every_n_iters": trainer_config.get("val_every_n_epoch", None),
                "callbacks": callbacks,
            },
            vessel_kwargs={
                "episode_per_iter": trainer_config["episode_per_collect"],
                "update_kwargs": {
                    "batch_size": trainer_config["batch_size"],
                    "repeat": trainer_config["repeat_per_collect"],
                },
                "val_initial_states": valid_dataset,
            },
        )

    if run_backtest:
        test_dataset = LazyLoadDataset(
            data_dir=data_config["source"]["feature_root_dir"],
            order_file_path=order_root_path / "test",
            default_start_time_index=data_config["source"]["default_start_time_index"] // data_granularity,
            default_end_time_index=data_config["source"]["default_end_time_index"] // data_granularity,
        )

        backtest(
            simulator_fn=_simulator_factory_simple,
            state_interpreter=state_interpreter,
            action_interpreter=action_interpreter,
            initial_states=test_dataset,
            policy=policy,
            logger=CsvWriter(Path(trainer_config["checkpoint_path"])),
            reward=reward,
            finite_env_type=env_config["parallel_mode"],
            concurrency=env_config["concurrency"],
        )


def main(config: dict, run_training: bool, run_backtest: bool) -> None:
    if not run_training and not run_backtest:
        warnings.warn("Skip the entire job since training and backtest are both skipped.")
        return

    if "seed" in config["runtime"]:
        seed_everything(config["runtime"]["seed"])

    state_interpreter: StateInterpreter = init_instance_by_config(config["state_interpreter"])
    action_interpreter: ActionInterpreter = init_instance_by_config(config["action_interpreter"])
    reward: Reward = init_instance_by_config(config["reward"])

    additional_policy_kwargs = {
        "obs_space": state_interpreter.observation_space,
        "action_space": action_interpreter.action_space,
    }

    # Create torch network
    if "network" in config:
        if "kwargs" not in config["network"]:
            config["network"]["kwargs"] = {}
        config["network"]["kwargs"].update({"obs_space": state_interpreter.observation_space})
        additional_policy_kwargs["network"] = init_instance_by_config(config["network"])

    # Create policy
    if "kwargs" not in config["policy"]:
        config["policy"]["kwargs"] = {}
    config["policy"]["kwargs"].update(additional_policy_kwargs)
    policy: BasePolicy = init_instance_by_config(config["policy"])

    use_cuda = config["runtime"].get("use_cuda", False)
    if use_cuda:
        policy.cuda()

    train_and_test(
        env_config=config["env"],
        simulator_config=config["simulator"],
        data_config=config["data"],
        trainer_config=config["trainer"],
        action_interpreter=action_interpreter,
        state_interpreter=state_interpreter,
        policy=policy,
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

    with open(args.config_path, "r") as input_stream:
        config = yaml.safe_load(input_stream)

    main(config, run_training=not args.no_training, run_backtest=args.run_backtest)
