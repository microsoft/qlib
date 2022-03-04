# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import warnings
from typing import Union

from ..log import get_module_logger
from ..utils import get_date_range
from ..utils.resam import Freq
from ..strategy.base import BaseStrategy
from ..backtest import get_exchange, position, backtest as backtest_func, executor as _executor


from ..data import D
from ..config import C
from ..data.dataset.utils import get_level_index


logger = get_module_logger("Evaluate")


def risk_analysis(r, N: int = None, freq: str = "day"):
    """Risk Analysis

    Parameters
    ----------
    r : pandas.Series
        daily return series.
    N: int
        scaler for annualizing information_ratio (day: 252, week: 50, month: 12), at least one of `N` and `freq` should exist
    freq: str
        analysis frequency used for calculating the scaler, at least one of `N` and `freq` should exist
    """

    def cal_risk_analysis_scaler(freq):
        _count, _freq = Freq.parse(freq)
        # len(D.calendar(start_time='2010-01-01', end_time='2019-12-31', freq='day')) = 2384
        _freq_scaler = {
            Freq.NORM_FREQ_MINUTE: 240 * 238,
            Freq.NORM_FREQ_DAY: 238,
            Freq.NORM_FREQ_WEEK: 50,
            Freq.NORM_FREQ_MONTH: 12,
        }
        return _freq_scaler[_freq] / _count

    if N is None and freq is None:
        raise ValueError("at least one of `N` and `freq` should exist")
    if N is not None and freq is not None:
        warnings.warn("risk_analysis freq will be ignored")
    if N is None:
        N = cal_risk_analysis_scaler(freq)

    mean = r.mean()
    std = r.std(ddof=1)
    annualized_return = mean * N
    information_ratio = mean / std * np.sqrt(N)
    max_drawdown = (r.cumsum() - r.cumsum().cummax()).min()
    data = {
        "mean": mean,
        "std": std,
        "annualized_return": annualized_return,
        "information_ratio": information_ratio,
        "max_drawdown": max_drawdown,
    }
    res = pd.Series(data).to_frame("risk")
    return res


def indicator_analysis(df, method="mean"):
    """analyze statistical time-series indicators of trading

    Parameters
    ----------
    df : pandas.DataFrame
        columns: like ['pa', 'pos', 'ffr', 'deal_amount', 'value'].
            Necessary fields:
                - 'pa' is the price advantage in trade indicators
                - 'pos' is the positive rate in trade indicators
                - 'ffr' is the fulfill rate in trade indicators
            Optional fields:
                - 'deal_amount' is the total deal deal_amount, only necessary when method is 'amount_weighted'
                - 'value' is the total trade value, only necessary when method is 'value_weighted'

        index: Index(datetime)
    method : str, optional
        statistics method of pa/ffr, by default "mean"
        - if method is 'mean', count the mean statistical value of each trade indicator
        - if method is 'amount_weighted', count the deal_amount weighted mean statistical value of each trade indicator
        - if method is 'value_weighted', count the value weighted mean statistical value of each trade indicator
        Note: statistics method of pos is always "mean"

    Returns
    -------
    pd.DataFrame
        statistical value of each trade indicators
    """
    weights_dict = {
        "mean": df["count"],
        "amount_weighted": df["deal_amount"].abs(),
        "value_weighted": df["value"].abs(),
    }
    if method not in weights_dict:
        raise ValueError(f"indicator_analysis method {method} is not supported!")

    # statistic pa/ffr indicator
    indicators_df = df[["ffr", "pa"]]
    weights = weights_dict.get(method)
    res = indicators_df.mul(weights, axis=0).sum() / weights.sum()

    # statistic pos
    weights = weights_dict.get("mean")
    res.loc["pos"] = df["pos"].mul(weights).sum() / weights.sum()
    res = res.to_frame("value")
    return res


# This is the API for compatibility for legacy code
def backtest_daily(
    start_time: Union[str, pd.Timestamp],
    end_time: Union[str, pd.Timestamp],
    strategy: Union[str, dict, BaseStrategy],
    executor: Union[str, dict, _executor.BaseExecutor] = None,
    account: Union[float, int, position.Position] = 1e8,
    benchmark: str = "SH000300",
    exchange_kwargs: dict = None,
    pos_type: str = "Position",
):
    """initialize the strategy and executor, then executor the backtest of daily frequency

    Parameters
    ----------
    start_time : Union[str, pd.Timestamp]
        closed start time for backtest
        **NOTE**: This will be applied to the outmost executor's calendar.
    end_time : Union[str, pd.Timestamp]
        closed end time for backtest
        **NOTE**: This will be applied to the outmost executor's calendar.
        E.g. Executor[day](Executor[1min]),   setting `end_time == 20XX0301` will include all the minutes on 20XX0301
    strategy : Union[str, dict, BaseStrategy]
        for initializing outermost portfolio strategy. Please refer to the docs of init_instance_by_config for more information.

        E.g.

        .. code-block:: python
            # dict
            strategy = {
                "class": "TopkDropoutStrategy",
                "module_path": "qlib.contrib.strategy.signal_strategy",
                "kwargs": {
                    "signal": (model, dataset),
                    "topk": 50,
                    "n_drop": 5,
                },
            }
            # BaseStrategy
            pred_score = pd.read_pickle("score.pkl")["score"]
            STRATEGY_CONFIG = {
                "topk": 50,
                "n_drop": 5,
                "signal": pred_score,
            }
            strategy = TopkDropoutStrategy(**STRATEGY_CONFIG)
            # str example.
            # 1) specify a pickle object
            #     - path like 'file:///<path to pickle file>/obj.pkl'
            # 2) specify a class name
            #     - "ClassName":  getattr(module, "ClassName")() will be used.
            # 3) specify module path with class name
            #     - "a.b.c.ClassName" getattr(<a.b.c.module>, "ClassName")() will be used.


    executor : Union[str, dict, BaseExecutor]
        for initializing the outermost executor.
    benchmark: str
        the benchmark for reporting.
    account : Union[float, int, Position]
        information for describing how to creating the account
        For `float` or `int`:
            Using Account with only initial cash
        For `Position`:
            Using Account with a Position
    exchange_kwargs : dict
        the kwargs for initializing Exchange
        E.g.

        .. code-block:: python

            exchange_kwargs = {
                "freq": freq,
                "limit_threshold": None, # limit_threshold is None, using C.limit_threshold
                "deal_price": None, # deal_price is None, using C.deal_price
                "open_cost": 0.0005,
                "close_cost": 0.0015,
                "min_cost": 5,
            }

    pos_type : str
        the type of Position.

    Returns
    -------
    report_normal: pd.DataFrame
        backtest report
    positions_normal: pd.DataFrame
        backtest positions

    """
    freq = "day"
    if executor is None:
        executor_config = {
            "time_per_step": freq,
            "generate_portfolio_metrics": True,
        }
        executor = _executor.SimulatorExecutor(**executor_config)
    _exchange_kwargs = {
        "freq": freq,
        "limit_threshold": None,
        "deal_price": None,
        "open_cost": 0.0005,
        "close_cost": 0.0015,
        "min_cost": 5,
    }
    if exchange_kwargs is not None:
        _exchange_kwargs.update(exchange_kwargs)

    portfolio_metric_dict, indicator_dict = backtest_func(
        start_time=start_time,
        end_time=end_time,
        strategy=strategy,
        executor=executor,
        account=account,
        benchmark=benchmark,
        exchange_kwargs=_exchange_kwargs,
        pos_type=pos_type,
    )
    analysis_freq = "{0}{1}".format(*Freq.parse(freq))

    report_normal, positions_normal = portfolio_metric_dict.get(analysis_freq)

    return report_normal, positions_normal


def long_short_backtest(
    pred,
    topk=50,
    deal_price=None,
    shift=1,
    open_cost=0,
    close_cost=0,
    trade_unit=None,
    limit_threshold=None,
    min_cost=5,
    subscribe_fields=[],
    extract_codes=False,
):
    """
    A backtest for long-short strategy

    :param pred:        The trading signal produced on day `T`.
    :param topk:       The short topk securities and long topk securities.
    :param deal_price:  The price to deal the trading.
    :param shift:       Whether to shift prediction by one day.  The trading day will be T+1 if shift==1.
    :param open_cost:   open transaction cost.
    :param close_cost:  close transaction cost.
    :param trade_unit:  100 for China A.
    :param limit_threshold: limit move 0.1 (10%) for example, long and short with same limit.
    :param min_cost:    min transaction cost.
    :param subscribe_fields: subscribe fields.
    :param extract_codes:  bool.
                       will we pass the codes extracted from the pred to the exchange.
                       NOTE: This will be faster with offline qlib.
    :return:            The result of backtest, it is represented by a dict.
                        { "long": long_returns(excess),
                          "short": short_returns(excess),
                          "long_short": long_short_returns}
    """
    if get_level_index(pred, level="datetime") == 1:
        pred = pred.swaplevel().sort_index()

    if trade_unit is None:
        trade_unit = C.trade_unit
    if limit_threshold is None:
        limit_threshold = C.limit_threshold
    if deal_price is None:
        deal_price = C.deal_price
    if deal_price[0] != "$":
        deal_price = "$" + deal_price

    subscribe_fields = subscribe_fields.copy()
    profit_str = f"Ref({deal_price}, -1)/{deal_price} - 1"
    subscribe_fields.append(profit_str)

    trade_exchange = get_exchange(
        pred=pred,
        deal_price=deal_price,
        subscribe_fields=subscribe_fields,
        limit_threshold=limit_threshold,
        open_cost=open_cost,
        close_cost=close_cost,
        min_cost=min_cost,
        trade_unit=trade_unit,
        extract_codes=extract_codes,
        shift=shift,
    )

    _pred_dates = pred.index.get_level_values(level="datetime")
    predict_dates = D.calendar(start_time=_pred_dates.min(), end_time=_pred_dates.max())
    trade_dates = np.append(predict_dates[shift:], get_date_range(predict_dates[-1], left_shift=1, right_shift=shift))

    long_returns = {}
    short_returns = {}
    ls_returns = {}

    for pdate, date in zip(predict_dates, trade_dates):
        score = pred.loc(axis=0)[pdate, :]
        score = score.reset_index().sort_values(by="score", ascending=False)

        long_stocks = list(score.iloc[:topk]["instrument"])
        short_stocks = list(score.iloc[-topk:]["instrument"])

        score = score.set_index(["datetime", "instrument"]).sort_index()

        long_profit = []
        short_profit = []
        all_profit = []

        for stock in long_stocks:
            if not trade_exchange.is_stock_tradable(stock_id=stock, trade_date=date):
                continue
            profit = trade_exchange.get_quote_info(stock_id=stock, trade_date=date)[profit_str]
            if np.isnan(profit):
                long_profit.append(0)
            else:
                long_profit.append(profit)

        for stock in short_stocks:
            if not trade_exchange.is_stock_tradable(stock_id=stock, trade_date=date):
                continue
            profit = trade_exchange.get_quote_info(stock_id=stock, trade_date=date)[profit_str]
            if np.isnan(profit):
                short_profit.append(0)
            else:
                short_profit.append(-profit)

        for stock in list(score.loc(axis=0)[pdate, :].index.get_level_values(level=0)):
            # exclude the suspend stock
            if trade_exchange.check_stock_suspended(stock_id=stock, trade_date=date):
                continue
            profit = trade_exchange.get_quote_info(stock_id=stock, trade_date=date)[profit_str]
            if np.isnan(profit):
                all_profit.append(0)
            else:
                all_profit.append(profit)

        long_returns[date] = np.mean(long_profit) - np.mean(all_profit)
        short_returns[date] = np.mean(short_profit) + np.mean(all_profit)
        ls_returns[date] = np.mean(short_profit) + np.mean(long_profit)

    return dict(
        zip(
            ["long", "short", "long_short"],
            map(pd.Series, [long_returns, short_returns, ls_returns]),
        )
    )


def t_run():
    pred_FN = "./check_pred.csv"
    pred: pd.DataFrame = pd.read_csv(pred_FN)
    pred["datetime"] = pd.to_datetime(pred["datetime"])
    pred = pred.set_index([pred.columns[0], pred.columns[1]])
    pred = pred.iloc[:9000]
    strategy_config = {
        "topk": 50,
        "n_drop": 5,
        "signal": pred,
    }
    report_df, positions = backtest_daily(start_time="2017-01-01", end_time="2020-08-01", strategy=strategy_config)
    print(report_df.head())
    print(positions.keys())
    print(positions[list(positions.keys())[0]])
    return 0


if __name__ == "__main__":
    t_run()
