# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .order import Order
from .account import Account
from .position import Position
from .exchange import Exchange
from .report import Report
from .backtest import backtest as backtest_func, get_date_range

import numpy as np
import inspect
from ...utils import init_instance_by_config
from ...log import get_module_logger
from ...config import C

logger = get_module_logger("backtest caller")


def get_strategy(
    strategy=None,
    topk=50,
    margin=0.5,
    n_drop=5,
    risk_degree=0.95,
    str_type="dropout",
    adjust_dates=None,
):
    """get_strategy

    There will be 3 ways to return a stratgy. Please follow the code.


    Parameters
    ----------

    strategy : Strategy()
        strategy used in backtest.
    topk : int (Default value: 50)
        top-N stocks to buy.
    margin : int or float(Default value: 0.5)
        - if isinstance(margin, int):

            sell_limit = margin

        - else:

            sell_limit = pred_in_a_day.count() * margin

        buffer margin, in single score_mode, continue holding stock if it is in nlargest(sell_limit).
        sell_limit should be no less than topk.
    n_drop : int
        number of stocks to be replaced in each trading date.
    risk_degree: float
        0-1, 0.95 for example, use 95% money to trade.
    str_type: 'amount', 'weight' or 'dropout'
        strategy type: TopkAmountStrategy ,TopkWeightStrategy or TopkDropoutStrategy.

    Returns
    -------
    :class: Strategy
    an initialized strategy object
    """

    # There  will be 3 ways to return a strategy.
    if strategy is None:
        # 1) create strategy with param `strategy`
        str_cls_dict = {
            "amount": "TopkAmountStrategy",
            "weight": "TopkWeightStrategy",
            "dropout": "TopkDropoutStrategy",
        }
        logger.info("Create new strategy ")
        from .. import strategy as strategy_pool

        str_cls = getattr(strategy_pool, str_cls_dict.get(str_type))
        strategy = str_cls(
            topk=topk,
            buffer_margin=margin,
            n_drop=n_drop,
            risk_degree=risk_degree,
            adjust_dates=adjust_dates,
        )
    elif isinstance(strategy, (dict, str)):
        # 2) create strategy with init_instance_by_config
        logger.info("Create new strategy ")
        strategy = init_instance_by_config(strategy)

    from ..strategy.strategy import BaseStrategy

    # else: nothing happens. 3) Use the strategy directly
    if not isinstance(strategy, BaseStrategy):
        raise TypeError("Strategy not supported")
    return strategy


def get_exchange(
    pred,
    exchange=None,
    subscribe_fields=[],
    open_cost=0.0015,
    close_cost=0.0025,
    min_cost=5.0,
    trade_unit=None,
    limit_threshold=None,
    deal_price=None,
    extract_codes=False,
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
    extract_codes: bool
        will we pass the codes extracted from the pred to the exchange.
        NOTE: This will be faster with offline qlib.

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
        if extract_codes:
            codes = sorted(pred.index.get_level_values("instrument").unique())
        else:
            codes = "all"  # TODO: We must ensure that 'all.txt' includes all the stocks

        dates = sorted(pred.index.get_level_values("datetime").unique())
        dates = np.append(dates, get_date_range(dates[-1], left_shift=1, right_shift=shift))

        exchange = Exchange(
            trade_dates=dates,
            codes=codes,
            deal_price=deal_price,
            subscribe_fields=subscribe_fields,
            limit_threshold=limit_threshold,
            open_cost=open_cost,
            close_cost=close_cost,
            min_cost=min_cost,
            trade_unit=trade_unit,
        )
    return exchange


def get_executor(
    executor=None,
    trade_exchange=None,
    verbose=True,
):
    """get_executor

    There will be 3 ways to return a executor. Please follow the code.

    Parameters
    ----------

    executor : BaseExecutor
        executor used in backtest.
    trade_exchange : Exchange
        exchange used in executor
    verbose : bool
        whether to print log.

    Returns
    -------
    :class: BaseExecutor
    an initialized BaseExecutor object
    """

    # There  will be 3 ways to return a executor.
    if executor is None:
        # 1) create executor with param `executor`
        logger.info("Create new executor ")
        from ..online.executor import SimulatorExecutor

        executor = SimulatorExecutor(trade_exchange=trade_exchange, verbose=verbose)
    elif isinstance(executor, (dict, str)):
        # 2) create executor with config
        logger.info("Create new executor ")
        executor = init_instance_by_config(executor)

    from ..online.executor import BaseExecutor

    # 3) Use the executor directly
    if not isinstance(executor, BaseExecutor):
        raise TypeError("Executor not supported")
    return executor


# This is the API for compatibility for legacy code
def backtest(pred, account=1e9, shift=1, benchmark="SH000905", verbose=True, return_order=False, **kwargs):
    """This function will help you set a reasonable Exchange and provide default value for strategy
    Parameters
    ----------

    - **backtest workflow related or commmon arguments**

    pred : pandas.DataFrame
        predict should has <datetime, instrument> index and one `score` column.
    account : float
        init account value.
    shift : int
        whether to shift prediction by one day.
    benchmark : str
        benchmark code, default is SH000905 CSI 500.
    verbose : bool
        whether to print log.
    return_order : bool
        whether to return order list

    - **strategy related arguments**

    strategy : Strategy()
        strategy used in backtest.
    topk : int (Default value: 50)
        top-N stocks to buy.
    margin : int or float(Default value: 0.5)
        - if isinstance(margin, int):

            sell_limit = margin

        - else:

            sell_limit = pred_in_a_day.count() * margin

        buffer margin, in single score_mode, continue holding stock if it is in nlargest(sell_limit).
        sell_limit should be no less than topk.
    n_drop : int
        number of stocks to be replaced in each trading date.
    risk_degree: float
        0-1, 0.95 for example, use 95% money to trade.
    str_type: 'amount', 'weight' or 'dropout'
        strategy type: TopkAmountStrategy ,TopkWeightStrategy or TopkDropoutStrategy.

    - **exchange related arguments**

    exchange: Exchange()
        pass the exchange for speeding up.
    subscribe_fields: list
        subscribe fields.
    open_cost : float
        open transaction cost. The default value is 0.002(0.2%).
    close_cost : float
        close transaction cost. The default value is 0.002(0.2%).
    min_cost : float
        min transaction cost.
    trade_unit : int
        100 for China A.
    deal_price: str
        dealing price type: 'close', 'open', 'vwap'.
    limit_threshold : float
        limit move 0.1 (10%) for example, long and short with same limit.
    extract_codes: bool
        will we pass the codes extracted from the pred to the exchange.

        .. note:: This will be faster with offline qlib.

    - **executor related arguments**

    executor : BaseExecutor()
        executor used in backtest.
    verbose : bool
        whether to print log.

    """
    # check strategy:
    spec = inspect.getfullargspec(get_strategy)
    str_args = {k: v for k, v in kwargs.items() if k in spec.args}
    strategy = get_strategy(**str_args)

    # init exchange:
    spec = inspect.getfullargspec(get_exchange)
    ex_args = {k: v for k, v in kwargs.items() if k in spec.args}
    trade_exchange = get_exchange(pred, **ex_args)

    # init executor:
    executor = get_executor(executor=kwargs.get("executor"), trade_exchange=trade_exchange, verbose=verbose)

    # run backtest
    report_dict = backtest_func(
        pred=pred,
        strategy=strategy,
        executor=executor,
        trade_exchange=trade_exchange,
        shift=shift,
        verbose=verbose,
        account=account,
        benchmark=benchmark,
        return_order=return_order,
    )
    # for  compatibility of the old API. return the dict positions

    positions = report_dict.get("positions")
    report_dict.update({"positions": {k: p.position for k, p in positions.items()}})
    return report_dict
