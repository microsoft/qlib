# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import argparse
import copy
import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed

from qlib.backtest import INDICATOR_METRIC, collect_data_loop, get_strategy_executor
from qlib.backtest.decision import BaseTradeDecision, Order, OrderDir, TradeRangeByTime
from qlib.backtest.executor import SimulatorExecutor
from qlib.backtest.high_performance_ds import BaseOrderIndicator
from qlib.rl.contrib.naive_config_parser import get_backtest_config_fromfile
from qlib.rl.contrib.utils import read_order_file
from qlib.rl.data.integration import init_qlib
from qlib.rl.order_execution.simulator_qlib import SingleAssetOrderExecution
from qlib.typehint import Literal


def _get_multi_level_executor_config(
    strategy_config: dict,
    cash_limit: float | None = None,
    generate_report: bool = False,
) -> dict:
    executor_config = {
        "class": "SimulatorExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {
            "time_per_step": "5min",  # FIXME: move this into config
            "verbose": False,
            "trade_type": SimulatorExecutor.TT_PARAL if cash_limit is not None else SimulatorExecutor.TT_SERIAL,
            "generate_report": generate_report,
            "track_data": True,
        },
    }

    freqs = list(strategy_config.keys())
    freqs.sort(key=pd.Timedelta)
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


def _convert_indicator_to_dataframe(indicator: dict) -> Optional[pd.DataFrame]:
    record_list = []
    for time, value_dict in indicator.items():
        if isinstance(value_dict, BaseOrderIndicator):
            # HACK: for qlib v0.8
            value_dict = value_dict.to_series()
        try:
            value_dict = copy.deepcopy(value_dict)
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


def _generate_report(
    decisions: List[BaseTradeDecision],
    report_indicators: List[INDICATOR_METRIC],
) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    """Generate backtest reports

    Parameters
    ----------
    decisions:
        List of trade decisions.
    report_indicators
        List of indicator reports.
    Returns
    -------

    """
    indicator_dict: Dict[str, List[pd.DataFrame]] = defaultdict(list)
    indicator_his: Dict[str, List[dict]] = defaultdict(list)

    for report_indicator in report_indicators:
        for key, (indicator_df, indicator_obj) in report_indicator.items():
            indicator_dict[key].append(indicator_df)
            indicator_his[key].append(indicator_obj.order_indicator_his)

    report = {}
    decision_details = pd.concat([getattr(d, "details") for d in decisions if hasattr(d, "details")])
    for key in indicator_dict:
        cur_dict = pd.concat(indicator_dict[key])
        cur_his = pd.concat([_convert_indicator_to_dataframe(his) for his in indicator_his[key]])
        cur_details = decision_details[decision_details.freq == key].set_index(["instrument", "datetime"])
        if len(cur_details) > 0:
            cur_details.pop("freq")
            cur_his = cur_his.join(cur_details, how="outer")

        report[key] = (cur_dict, cur_his)

    return report


def single_with_simulator(
    backtest_config: dict,
    orders: pd.DataFrame,
    split: Literal["stock", "day"] = "stock",
    cash_limit: float | None = None,
    generate_report: bool = False,
) -> Union[Tuple[pd.DataFrame, dict], pd.DataFrame]:
    """Run backtest in a single thread with SingleAssetOrderExecution simulator. The orders will be executed day by day.
    A new simulator will be created and used for every single-day order.

    Parameters
    ----------
    backtest_config:
        Backtest config
    orders:
        Orders to be executed. Example format:
                 datetime instrument  amount  direction
            0  2020-06-01       INST   600.0          0
            1  2020-06-02       INST   700.0          1
            ...
    split
        Method to split orders. If it is "stock", split orders by stock. If it is "day", split orders by date.
    cash_limit
        Limitation of cash.
    generate_report
        Whether to generate reports.

    Returns
    -------
        If generate_report is True, return execution records and the generated report. Otherwise, return only records.
    """
    init_qlib(backtest_config["qlib"])

    stocks = orders.instrument.unique().tolist()

    reports = []
    decisions = []
    for _, row in orders.iterrows():
        date = pd.Timestamp(row["datetime"])
        start_time = pd.Timestamp(backtest_config["start_time"]).replace(year=date.year, month=date.month, day=date.day)
        end_time = pd.Timestamp(backtest_config["end_time"]).replace(year=date.year, month=date.month, day=date.day)
        order = Order(
            stock_id=row["instrument"],
            amount=row["amount"],
            direction=OrderDir(row["direction"]),
            start_time=start_time,
            end_time=end_time,
        )

        executor_config = _get_multi_level_executor_config(
            strategy_config=backtest_config["strategies"],
            cash_limit=cash_limit,
            generate_report=generate_report,
        )

        exchange_config = copy.deepcopy(backtest_config["exchange"])
        exchange_config.update(
            {
                "codes": stocks,
                "freq": "5min",  # FIXME: move this into config
            }
        )

        simulator = SingleAssetOrderExecution(
            order=order,
            executor_config=executor_config,
            exchange_config=exchange_config,
            qlib_config=None,
            cash_limit=None,
        )

        reports.append(simulator.report_dict)
        decisions += simulator.decisions

    indicator_1day_objs = [report["indicator"]["1day"][1] for report in reports]
    indicator_info = {k: v for obj in indicator_1day_objs for k, v in obj.order_indicator_his.items()}
    records = _convert_indicator_to_dataframe(indicator_info)
    assert records is None or not np.isnan(records["ffr"]).any()

    if generate_report:
        _report = _generate_report(decisions, [report["indicator"] for report in reports])

        if split == "stock":
            stock_id = orders.iloc[0].instrument
            report = {stock_id: _report}
        else:
            day = orders.iloc[0].datetime
            report = {day: _report}

        return records, report
    else:
        return records


def single_with_collect_data_loop(
    backtest_config: dict,
    orders: pd.DataFrame,
    split: Literal["stock", "day"] = "stock",
    cash_limit: float | None = None,
    generate_report: bool = False,
) -> Union[Tuple[pd.DataFrame, dict], pd.DataFrame]:
    """Run backtest in a single thread with collect_data_loop.

    Parameters
    ----------
    backtest_config:
        Backtest config
    orders:
        Orders to be executed. Example format:
                 datetime instrument  amount  direction
            0  2020-06-01       INST   600.0          0
            1  2020-06-02       INST   700.0          1
            ...
    split
        Method to split orders. If it is "stock", split orders by stock. If it is "day", split orders by date.
    cash_limit
        Limitation of cash.
    generate_report
        Whether to generate reports.

    Returns
    -------
        If generate_report is True, return execution records and the generated report. Otherwise, return only records.
    """

    init_qlib(backtest_config["qlib"])

    trade_start_time = orders["datetime"].min()
    trade_end_time = orders["datetime"].max()
    stocks = orders.instrument.unique().tolist()

    strategy_config = {
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

    executor_config = _get_multi_level_executor_config(
        strategy_config=backtest_config["strategies"],
        cash_limit=cash_limit,
        generate_report=generate_report,
    )

    exchange_config = copy.deepcopy(backtest_config["exchange"])
    exchange_config.update(
        {
            "codes": stocks,
            "freq": "5min",  # FIXME: move this into config
        }
    )

    strategy, executor = get_strategy_executor(
        start_time=pd.Timestamp(trade_start_time),
        end_time=pd.Timestamp(trade_end_time) + pd.DateOffset(1),
        strategy=strategy_config,
        executor=executor_config,
        benchmark=None,
        account=cash_limit if cash_limit is not None else int(1e12),
        exchange_kwargs=exchange_config,
        pos_type="Position" if cash_limit is not None else "InfPosition",
    )

    report_dict: dict = {}
    decisions = list(collect_data_loop(trade_start_time, trade_end_time, strategy, executor, report_dict))

    indicator_dict = cast(INDICATOR_METRIC, report_dict.get("indicator_dict"))
    records = _convert_indicator_to_dataframe(indicator_dict["1day"][1].order_indicator_his)
    assert records is None or not np.isnan(records["ffr"]).any()

    if generate_report:
        _report = _generate_report(decisions, [indicator_dict])
        if split == "stock":
            stock_id = orders.iloc[0].instrument
            report = {stock_id: _report}
        else:
            day = orders.iloc[0].datetime
            report = {day: _report}
        return records, report
    else:
        return records


def backtest(backtest_config: dict, with_simulator: bool = False) -> pd.DataFrame:
    order_df = read_order_file(backtest_config["order_file"])

    cash_limit = backtest_config["exchange"].pop("cash_limit")
    generate_report = backtest_config.pop("generate_report")

    stock_pool = order_df["instrument"].unique().tolist()
    stock_pool.sort()

    single = single_with_simulator if with_simulator else single_with_collect_data_loop
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

    if not output_path.exists():
        os.makedirs(output_path)

    if "pa" in res.columns:
        res["pa"] = res["pa"] * 10000.0  # align with training metrics
    res.to_csv(output_path / "backtest_result.csv")
    return res


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True, help="Path to the config file")
    parser.add_argument("--use_simulator", action="store_true", help="Whether to use simulator as the backend")
    parser.add_argument(
        "--n_jobs",
        type=int,
        required=False,
        help="The number of jobs for running backtest parallely(1 for single process)",
    )
    args = parser.parse_args()

    config = get_backtest_config_fromfile(args.config_path)
    if args.n_jobs is not None:
        config["concurrency"] = args.n_jobs

    backtest(
        backtest_config=config,
        with_simulator=args.use_simulator,
    )
