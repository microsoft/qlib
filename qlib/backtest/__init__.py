# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import copy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generator, List, Optional, Tuple, Union

import pandas as pd

from .account import Account

if TYPE_CHECKING:
    from ..strategy.base import BaseStrategy
    from .executor import BaseExecutor
    from .decision import BaseTradeDecision

from ..config import C
from ..log import get_module_logger
from ..utils import init_instance_by_config
from .backtest import INDICATOR_METRIC, PORT_METRIC, backtest_loop, collect_data_loop
from .decision import Order
from .exchange import Exchange
from .utils import CommonInfrastructure

# make import more user-friendly by adding `from qlib.backtest import STH`


logger = get_module_logger("backtest caller")


def get_exchange(
    exchange: Union[str, dict, object, Path] = None,
    freq: str = "day",
    start_time: Union[pd.Timestamp, str] = None,
    end_time: Union[pd.Timestamp, str] = None,
    codes: Union[list, str] = "all",
    subscribe_fields: list = [],
    open_cost: float = 0.0015,
    close_cost: float = 0.0025,
    min_cost: float = 5.0,
    limit_threshold: Union[Tuple[str, str], float, None] = None,
    deal_price: Union[str, Tuple[str, str], List[str]] = None,
    **kwargs: Any,
) -> Exchange:
    """get_exchange

    Parameters
    ----------

    # exchange related arguments
    exchange: Exchange
        It could be None or any types that are acceptable by `init_instance_by_config`.
    freq: str
        frequency of data.
    start_time: Union[pd.Timestamp, str]
        closed start time for backtest.
    end_time: Union[pd.Timestamp, str]
        closed end time for backtest.
    codes: Union[list, str]
        list stock_id list or a string of instruments (i.e. all, csi500, sse50)
    subscribe_fields: list
        subscribe fields.
    open_cost : float
        open transaction cost. It is a ratio. The cost is proportional to your order's deal amount.
    close_cost : float
        close transaction cost. It is a ratio. The cost is proportional to your order's deal amount.
    min_cost : float
        min transaction cost.  It is an absolute amount of cost instead of a ratio of your order's deal amount.
        e.g. You must pay at least 5 yuan of commission regardless of your order's deal amount.
    deal_price: Union[str, Tuple[str, str], List[str]]
                The `deal_price` supports following two types of input
                - <deal_price> : str
                - (<buy_price>, <sell_price>): Tuple[str, str] or List[str]

                <deal_price>, <buy_price> or <sell_price> := <price>
                <price> := str
                - for example '$close', '$open', '$vwap' ("close" is OK. `Exchange` will help to prepend
                  "$" to the expression)
    limit_threshold : float
        limit move 0.1 (10%) for example, long and short with same limit.

    Returns
    -------
    :class: Exchange
    an initialized Exchange object
    """

    if limit_threshold is None:
        limit_threshold = C.limit_threshold
    if exchange is None:
        logger.info("Create new exchange")

        exchange = Exchange(
            freq=freq,
            start_time=start_time,
            end_time=end_time,
            codes=codes,
            deal_price=deal_price,
            subscribe_fields=subscribe_fields,
            limit_threshold=limit_threshold,
            open_cost=open_cost,
            close_cost=close_cost,
            min_cost=min_cost,
            **kwargs,
        )
        return exchange
    else:
        return init_instance_by_config(exchange, accept_types=Exchange)


def create_account_instance(
    start_time: Union[pd.Timestamp, str],
    end_time: Union[pd.Timestamp, str],
    benchmark: Optional[str],
    account: Union[float, int, dict],
    pos_type: str = "Position",
) -> Account:
    """
    # TODO: is very strange pass benchmark_config in the account (maybe for report)
    # There should be a post-step to process the report.

    Parameters
    ----------
    start_time
        start time of the benchmark
    end_time
        end time of the benchmark
    benchmark : str
        the benchmark for reporting
    account :   Union[
                    float,
                    {
                        "cash": float,
                        "stock1": Union[
                                        int,    # it is equal to {"amount": int}
                                        {"amount": int, "price"(optional): float},
                                  ]
                    },
                ]
        information for describing how to creating the account
        For `float`:
            Using Account with only initial cash
        For `dict`:
            key "cash" means initial cash.
            key "stock1" means the information of first stock with amount and price(optional).
            ...
    pos_type: str
        Postion type.
    """
    if isinstance(account, (int, float)):
        init_cash = account
        position_dict = {}
    elif isinstance(account, dict):
        init_cash = account.pop("cash")
        position_dict = account
    else:
        raise ValueError("account must be in (int, float, dict)")

    return Account(
        init_cash=init_cash,
        position_dict=position_dict,
        pos_type=pos_type,
        benchmark_config={}
        if benchmark is None
        else {
            "benchmark": benchmark,
            "start_time": start_time,
            "end_time": end_time,
        },
    )


def get_strategy_executor(
    start_time: Union[pd.Timestamp, str],
    end_time: Union[pd.Timestamp, str],
    strategy: Union[str, dict, object, Path],
    executor: Union[str, dict, object, Path],
    benchmark: Optional[str] = "SH000300",
    account: Union[float, int, dict] = 1e9,
    exchange_kwargs: dict = {},
    pos_type: str = "Position",
) -> Tuple[BaseStrategy, BaseExecutor]:

    # NOTE:
    # - for avoiding recursive import
    # - typing annotations is not reliable
    from ..strategy.base import BaseStrategy  # pylint: disable=C0415
    from .executor import BaseExecutor  # pylint: disable=C0415

    trade_account = create_account_instance(
        start_time=start_time,
        end_time=end_time,
        benchmark=benchmark,
        account=account,
        pos_type=pos_type,
    )

    exchange_kwargs = copy.copy(exchange_kwargs)
    if "start_time" not in exchange_kwargs:
        exchange_kwargs["start_time"] = start_time
    if "end_time" not in exchange_kwargs:
        exchange_kwargs["end_time"] = end_time
    trade_exchange = get_exchange(**exchange_kwargs)

    common_infra = CommonInfrastructure(trade_account=trade_account, trade_exchange=trade_exchange)
    trade_strategy = init_instance_by_config(strategy, accept_types=BaseStrategy)
    trade_strategy.reset_common_infra(common_infra)
    trade_executor = init_instance_by_config(executor, accept_types=BaseExecutor)
    trade_executor.reset_common_infra(common_infra)

    return trade_strategy, trade_executor


def backtest(
    start_time: Union[pd.Timestamp, str],
    end_time: Union[pd.Timestamp, str],
    strategy: Union[str, dict, object, Path],
    executor: Union[str, dict, object, Path],
    benchmark: str = "SH000300",
    account: Union[float, int, dict] = 1e9,
    exchange_kwargs: dict = {},
    pos_type: str = "Position",
) -> Tuple[PORT_METRIC, INDICATOR_METRIC]:
    """initialize the strategy and executor, then backtest function for the interaction of the outermost strategy and
    executor in the nested decision execution

    Parameters
    ----------
    start_time : Union[pd.Timestamp, str]
        closed start time for backtest
        **NOTE**: This will be applied to the outmost executor's calendar.
    end_time : Union[pd.Timestamp, str]
        closed end time for backtest
        **NOTE**: This will be applied to the outmost executor's calendar.
        E.g. Executor[day](Executor[1min]),   setting `end_time == 20XX0301` will include all the minutes on 20XX0301
    strategy : Union[str, dict, object, Path]
        for initializing outermost portfolio strategy. Please refer to the docs of init_instance_by_config for more
        information.
    executor : Union[str, dict, object, Path]
        for initializing the outermost executor.
    benchmark: str
        the benchmark for reporting.
    account : Union[float, int, Position]
        information for describing how to create the account
        For `float` or `int`:
            Using Account with only initial cash
        For `Position`:
            Using Account with a Position
    exchange_kwargs : dict
        the kwargs for initializing Exchange
    pos_type : str
        the type of Position.

    Returns
    -------
    portfolio_dict: PORT_METRIC
        it records the trading portfolio_metrics information
    indicator_dict: INDICATOR_METRIC
        it computes the trading indicator
        It is organized in a dict format

    """
    trade_strategy, trade_executor = get_strategy_executor(
        start_time,
        end_time,
        strategy,
        executor,
        benchmark,
        account,
        exchange_kwargs,
        pos_type=pos_type,
    )
    return backtest_loop(start_time, end_time, trade_strategy, trade_executor)


def collect_data(
    start_time: Union[pd.Timestamp, str],
    end_time: Union[pd.Timestamp, str],
    strategy: Union[str, dict, object, Path],
    executor: Union[str, dict, object, Path],
    benchmark: str = "SH000300",
    account: Union[float, int, dict] = 1e9,
    exchange_kwargs: dict = {},
    pos_type: str = "Position",
    return_value: dict = None,
) -> Generator[object, None, None]:
    """initialize the strategy and executor, then collect the trade decision data for rl training

    please refer to the docs of the backtest for the explanation of the parameters

    Yields
    -------
    object
        trade decision
    """
    trade_strategy, trade_executor = get_strategy_executor(
        start_time,
        end_time,
        strategy,
        executor,
        benchmark,
        account,
        exchange_kwargs,
        pos_type=pos_type,
    )
    yield from collect_data_loop(start_time, end_time, trade_strategy, trade_executor, return_value=return_value)


def format_decisions(
    decisions: List[BaseTradeDecision],
) -> Optional[Tuple[str, List[Tuple[BaseTradeDecision, Union[Tuple, None]]]]]:
    """
    format the decisions collected by `qlib.backtest.collect_data`
    The decisions will be organized into a tree-like structure.

    Parameters
    ----------
    decisions : List[BaseTradeDecision]
        decisions collected by `qlib.backtest.collect_data`

    Returns
    -------
    Tuple[str, List[Tuple[BaseTradeDecision, Union[Tuple, None]]]]:

        reformat the list of decisions into a more user-friendly format
        <decisions> :=  Tuple[<freq>, List[Tuple[<decision>, <sub decisions>]]]
        - <sub decisions> := `<decisions> in lower level` | None
        - <freq> := "day" | "30min" | "1min" | ...
        - <decision> := <instance of BaseTradeDecision>
    """
    if len(decisions) == 0:
        return None

    cur_freq = decisions[0].strategy.trade_calendar.get_freq()

    res: Tuple[str, list] = (cur_freq, [])
    last_dec_idx = 0
    for i, dec in enumerate(decisions[1:], 1):
        if dec.strategy.trade_calendar.get_freq() == cur_freq:
            res[1].append((decisions[last_dec_idx], format_decisions(decisions[last_dec_idx + 1 : i])))
            last_dec_idx = i
    res[1].append((decisions[last_dec_idx], format_decisions(decisions[last_dec_idx + 1 :])))
    return res


__all__ = ["Order", "backtest", "get_strategy_executor"]
