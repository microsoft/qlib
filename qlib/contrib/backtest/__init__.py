# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .order import Order
from .position import Position
from .exchange import Exchange
from .report import Report
from .backtest import backtest as backtest_func

import copy
import numpy as np
import inspect
from ...utils import init_instance_by_config
from ...log import get_module_logger
from ...config import C

logger = get_module_logger("backtest caller")


def get_exchange(
    exchange=None,
    start_time=None,
    end_time=None,
    codes = "all",
    subscribe_fields=[],
    open_cost=0.0015,
    close_cost=0.0025,
    min_cost=5.0,
    trade_unit=None,
    limit_threshold=None,
    deal_price=None,
    shift=1,
):
    """get_exchange

    Parameters
    ----------

    # exchange related arguments
    exchange: Exchange().
    subscribe_fields: list
        subscribe fields.
    open_cost : float
        open transaction cost.
    close_cost : float
        close transaction cost.
    min_cost : float
        min transaction cost.
    trade_unit : int
        100 for China A.
    deal_price: str
        dealing price type: 'close', 'open', 'vwap'.
    limit_threshold : float
        limit move 0.1 (10%) for example, long and short with same limit.

    Returns
    -------
    :class: Exchange
    an initialized Exchange object
    """

    if trade_unit is None:
        trade_unit = C.trade_unit
    if limit_threshold is None:
        limit_threshold = C.limit_threshold
    if deal_price is None:
        deal_price = C.deal_price
    if exchange is None:
        logger.info("Create new exchange")
        # handle exception for deal_price
        if deal_price[0] != "$":
            deal_price = "$" + deal_price

        exchange = Exchange(
            start_time=start_time,
            end_time=end_time,
            codes=codes,
            deal_price=deal_price,
            subscribe_fields=subscribe_fields,
            limit_threshold=limit_threshold,
            open_cost=open_cost,
            close_cost=close_cost,
            trade_unit=trade_unit,
            min_cost=min_cost,
        )
        return exchange
    else:
        return init_instance_by_config(exchange, accept_types=Exchange)

def init_env_instance_by_config(env):
    if isinstance(env, dict):
        env_config = copy.copy(env)
        if "kwargs" in env_config:
            env_kwargs = copy.copy(env_config["kwargs"])
            if "sub_env" in env_kwargs:
                env_kwargs["sub_env"] = init_env_instance_by_config(env_kwargs["sub_env"])
            if "sub_strategy" in env_kwargs:
                env_kwargs["sub_strategy"] = init_instance_by_config(env_kwargs["sub_strategy"])
            env_config["kwargs"] = env_kwargs
        return init_instance_by_config(env_config)
    else:
        return env

def setup_exchange(root_instance, trade_exchange=None, force=False):
    if "trade_exchange" in inspect.getfullargspec(root_instance.__class__).args:
        if force:
            root_instance.reset(trade_exchange=trade_exchange)
        else:
            if not hasattr(root_instance, "trade_exchange") or root_instance.trade_exchange is None:
                root_instance.reset(trade_exchange=trade_exchange)
    if hasattr(root_instance, "sub_env"):
        setup_exchange(root_instance.sub_env, trade_exchange)
    if hasattr(root_instance, "sub_strategy"):
        setup_exchange(root_instance.sub_strategy, trade_exchange)
    
            
def backtest(start_time, end_time, strategy, env, benchmark=None, account=1e9, **kwargs):
    trade_strategy = init_instance_by_config(strategy)
    trade_env = init_env_instance_by_config(env)

    spec = inspect.getfullargspec(get_exchange)
    exchange_args = {k: v for k, v in kwargs.items() if k in spec.args}
    trade_exchange = get_exchange(**exchange_args)

    setup_exchange(trade_env, trade_exchange)
    setup_exchange(trade_strategy, trade_exchange)

    report_dict = backtest_func(start_time, end_time, trade_strategy, trade_env, benchmark, account)

    return report_dict
