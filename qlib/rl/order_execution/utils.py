# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from typing import Any, List, Tuple, cast

import numpy as np
import pandas as pd

from qlib.backtest import CommonInfrastructure, get_exchange
from qlib.backtest.account import Account
from qlib.backtest.decision import OrderDir
from qlib.backtest.executor import BaseExecutor
from qlib.rl.from_neutrader.config import ExchangeConfig
from qlib.rl.order_execution.simulator_simple import ONE_SEC, _float_or_ndarray
from qlib.utils.time import Freq


def get_common_infra(
    config: ExchangeConfig,
    trade_date: pd.Timestamp,
    codes: List[str],
    cash_limit: float = None,
) -> CommonInfrastructure:
    # need to specify a range here for acceleration
    if cash_limit is None:
        trade_account = Account(init_cash=int(1e12), benchmark_config={}, pos_type="InfPosition")
    else:
        trade_account = Account(
            init_cash=cash_limit,
            benchmark_config={},
            pos_type="Position",
            position_dict={code: {"amount": 1e12, "price": 1.0} for code in codes},
        )

    exchange = get_exchange(
        codes=codes,
        freq="1min",
        limit_threshold=config.limit_threshold,
        deal_price=config.deal_price,
        open_cost=config.open_cost,
        close_cost=config.close_cost,
        min_cost=config.min_cost if config.trade_unit is not None else 0,
        start_time=trade_date,
        end_time=trade_date + pd.DateOffset(1),
        trade_unit=config.trade_unit,
        volume_threshold=config.volume_threshold,
    )

    return CommonInfrastructure(trade_account=trade_account, trade_exchange=exchange)


def get_ticks_slice(
    ticks_index: pd.DatetimeIndex,
    start: pd.Timestamp,
    end: pd.Timestamp,
    include_end: bool = False,
) -> pd.DatetimeIndex:
    if not include_end:
        end = end - ONE_SEC
    return ticks_index[ticks_index.slice_indexer(start, end)]


def dataframe_append(df: pd.DataFrame, other: Any) -> pd.DataFrame:
    # dataframe.append is deprecated
    other_df = pd.DataFrame(other).set_index("datetime")
    other_df.index.name = "datetime"

    res = pd.concat([df, other_df], axis=0)
    return res


def price_advantage(
    exec_price: _float_or_ndarray,
    baseline_price: float,
    direction: OrderDir | int,
) -> _float_or_ndarray:
    if baseline_price == 0:  # something is wrong with data. Should be nan here
        if isinstance(exec_price, float):
            return 0.0
        else:
            return np.zeros_like(exec_price)
    if direction == OrderDir.BUY:
        res = (1 - exec_price / baseline_price) * 10000
    elif direction == OrderDir.SELL:
        res = (exec_price / baseline_price - 1) * 10000
    else:
        raise ValueError(f"Unexpected order direction: {direction}")
    res_wo_nan: np.ndarray = np.nan_to_num(res, nan=0.0)
    if res_wo_nan.size == 1:
        return res_wo_nan.item()
    else:
        return cast(_float_or_ndarray, res_wo_nan)


def get_portfolio_and_indicator(executor: BaseExecutor) -> Tuple[dict, dict]:
    all_executors = executor.get_all_executors()
    all_portfolio_metrics = {
        "{}{}".format(*Freq.parse(_executor.time_per_step)): _executor.trade_account.get_portfolio_metrics()
        for _executor in all_executors
        if _executor.trade_account.is_port_metr_enabled()
    }

    all_indicators = {}
    for _executor in all_executors:
        key = "{}{}".format(*Freq.parse(_executor.time_per_step))
        all_indicators[key] = _executor.trade_account.get_trade_indicator().generate_trade_indicators_dataframe()
        all_indicators[key + "_obj"] = _executor.trade_account.get_trade_indicator()

    return all_portfolio_metrics, all_indicators
