# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import copy
import pickle
import sys
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed

from qlib.backtest import collect_data_loop, get_strategy_executor
from qlib.backtest.decision import TradeRangeByTime
from qlib.backtest.executor import BaseExecutor, NestedExecutor, SimulatorExecutor
from qlib.backtest.high_performance_ds import BaseOrderIndicator
from qlib.rl.contrib.naive_config_parser import get_backtest_config_fromfile
from qlib.rl.contrib.utils import read_order_file
from qlib.rl.data.integration import init_qlib
from qlib.rl.utils.env_wrapper import CollectDataEnvWrapper


def _get_multi_level_executor_config(
    strategy_config: dict,
    cash_limit: float = None,
    generate_report: bool = False,
) -> dict:
    executor_config = {
        "class": "SimulatorExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {
            "time_per_step": "1min",
            "verbose": False,
            "trade_type": SimulatorExecutor.TT_PARAL if cash_limit is not None else SimulatorExecutor.TT_SERIAL,
            "generate_report": generate_report,
            "track_data": True,
        },
    }

    freqs = list(strategy_config.keys())
    freqs.sort(key=lambda x: pd.Timedelta(x))
    for freq in freqs:
        executor_config = {
            "class": "NestedExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {
                "time_per_step": freq,
                "inner_strategy": strategy_config[freq],
                "inner_executor": executor_config,
                "track_data": True,
            },
        }

    return executor_config


def _set_env_for_all_strategy(executor: BaseExecutor) -> None:
    if isinstance(executor, NestedExecutor):
        if hasattr(executor.inner_strategy, "set_env"):
            env = CollectDataEnvWrapper()
            env.reset()
            executor.inner_strategy.set_env(env)
        _set_env_for_all_strategy(executor.inner_executor)


def _convert_indicator_to_dataframe(indicator: dict) -> Optional[pd.DataFrame]:
    record_list = []
    for time, value_dict in indicator.items():
        if isinstance(value_dict, BaseOrderIndicator):
            # HACK: for qlib v0.8
            value_dict = value_dict.to_series()
        try:
            value_dict = {k: v for k, v in value_dict.items()}
            if value_dict["ffr"].empty:
                continue
        except Exception:
            value_dict = {k: v for k, v in value_dict.items() if k != "pa"}
        value_dict = pd.DataFrame(value_dict)
        value_dict["datetime"] = time
        record_list.append(value_dict)

    if not record_list:
        return None

    records: pd.DataFrame = pd.concat(record_list, 0).reset_index().rename(columns={"index": "instrument"})
    records = records.set_index(["instrument", "datetime"])
    return records


def _generate_report(decisions: list, report_dict: dict) -> dict:
    report = {}
    decision_details = pd.concat([d.details for d in decisions if hasattr(d, "details")])
    for key in ["1minute", "5minute", "30minute", "1day"]:
        if key not in report_dict["indicator"]:
            continue
        report[key] = report_dict["indicator"][key]
        report[key + "_obj"] = _convert_indicator_to_dataframe(
            report_dict["indicator"][key + "_obj"].order_indicator_his
        )
        cur_details = decision_details[decision_details.freq == key.rstrip("ute")].set_index(["instrument", "datetime"])
        if len(cur_details) > 0:
            cur_details.pop("freq")
            report[key + "_obj"] = report[key + "_obj"].join(cur_details, how="outer")
    if "1minute" in report_dict["report"]:
        report["simulator"] = report_dict["report"]["1minute"][0]
    return report


def single(
    backtest_config: dict,
    orders: pd.DataFrame,
    split: str = "stock",
    cash_limit: float = None,
    generate_report: bool = False,
) -> Union[Tuple[pd.DataFrame, dict], pd.DataFrame]:
    if split == "stock":
        stock_id = orders.iloc[0].instrument
        init_qlib(backtest_config["qlib"], part=stock_id)
    else:
        day = orders.iloc[0].datetime
        init_qlib(backtest_config["qlib"], part=day)

    trade_start_time = orders["datetime"].min()
    trade_end_time = orders["datetime"].max()
    stocks = orders.instrument.unique().tolist()

    top_strategy_config = {
        "class": "FileOrderStrategy",
        "module_path": "qlib.contrib.strategy.rule_strategy",
        "kwargs": {
            "file": orders,
            "trade_range": TradeRangeByTime(
                pd.Timestamp(backtest_config["start_time"]).time(),
                pd.Timestamp(backtest_config["end_time"]).time(),
            ),
        },
    }

    top_executor_config = _get_multi_level_executor_config(
        strategy_config=backtest_config["strategies"],
        cash_limit=cash_limit,
        generate_report=generate_report,
    )

    tmp_backtest_config = copy.deepcopy(backtest_config["exchange"])
    tmp_backtest_config.update(
        {
            "codes": stocks,
            "freq": "1min",
        }
    )

    strategy, executor = get_strategy_executor(
        start_time=pd.Timestamp(trade_start_time),
        end_time=pd.Timestamp(trade_end_time) + pd.DateOffset(1),
        strategy=top_strategy_config,
        executor=top_executor_config,
        benchmark=None,
        account=cash_limit if cash_limit is not None else int(1e12),
        exchange_kwargs=tmp_backtest_config,
        pos_type="Position" if cash_limit is not None else "InfPosition",
    )
    _set_env_for_all_strategy(executor=executor)

    report_dict: dict = {}
    decisions = list(collect_data_loop(trade_start_time, trade_end_time, strategy, executor, report_dict))

    records = _convert_indicator_to_dataframe(report_dict["indicator"]["1day_obj"].order_indicator_his)
    assert records is None or not np.isnan(records["ffr"]).any()

    if generate_report:
        report = _generate_report(decisions, report_dict)
        if split == "stock":
            stock_id = orders.iloc[0].instrument
            report = {stock_id: report}
        else:
            day = orders.iloc[0].datetime
            report = {day: report}
        return records, report
    else:
        return records


def backtest(backtest_config: dict) -> pd.DataFrame:
    order_df = read_order_file(backtest_config["order_file"])

    cash_limit = backtest_config["exchange"].pop("cash_limit")
    generate_report = backtest_config["exchange"].pop("generate_report")

    stock_pool = order_df["instrument"].unique().tolist()
    stock_pool.sort()

    mp_config = {"n_jobs": backtest_config["concurrency"], "verbose": 10, "backend": "multiprocessing"}
    torch.set_num_threads(1)  # https://github.com/pytorch/pytorch/issues/17199
    res = Parallel(**mp_config)(
        delayed(single)(
            backtest_config=backtest_config,
            orders=order_df[order_df["instrument"] == stock].copy(),
            split="stock",
            cash_limit=cash_limit,
            generate_report=generate_report,
        )
        for stock in stock_pool
    )

    output_path = Path(backtest_config["output_dir"])
    if generate_report:
        with (output_path / "report.pkl").open("wb") as f:
            report = {}
            for r in res:
                report.update(r[1])
            pickle.dump(report, f)
        res = pd.concat([r[0] for r in res], 0)
    else:
        res = pd.concat(res)

    res.to_csv(output_path / "summary.csv")
    return res


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    path = sys.argv[1]
    backtest(get_backtest_config_fromfile(path))
