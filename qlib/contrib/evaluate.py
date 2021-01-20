# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import warnings
from ..log import get_module_logger
from .backtest import get_exchange, backtest as backtest_func
from .backtest.backtest import get_date_range

from ..data import D
from ..config import C
from ..data.dataset.utils import get_level_index


logger = get_module_logger("Evaluate")


def risk_analysis(r, N=252):
    """Risk Analysis

    Parameters
    ----------
    r : pandas.Series
        daily return series.
    N: int
        scaler for annualizing information_ratio (day: 250, week: 50, month: 12).
    """
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
    res = pd.Series(data, index=data.keys()).to_frame("risk")
    return res


# This is the API for compatibility for legacy code
def backtest(pred, account=1e9, shift=1, benchmark="SH000905", verbose=True, **kwargs):
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
    warnings.warn(
        "this function is deprecated, please use backtest function in qlib.contrib.backtest", DeprecationWarning
    )
    report_dict = backtest_func(
        pred=pred, account=account, shift=shift, benchmark=benchmark, verbose=verbose, return_order=False, **kwargs
    )
    return report_dict.get("report_df"), report_dict.get("positions")


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
    pred = pd.read_csv(pred_FN)
    pred["datetime"] = pd.to_datetime(pred["datetime"])
    pred = pred.set_index([pred.columns[0], pred.columns[1]])
    pred = pred.iloc[:9000]
    report_df, positions = backtest(pred=pred)
    print(report_df.head())
    print(positions.keys())
    print(positions[list(positions.keys())[0]])
    return 0


if __name__ == "__main__":
    t_run()
