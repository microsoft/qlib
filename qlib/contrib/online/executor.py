# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import re
import json
import copy
import pathlib
import pandas as pd
from ...data import D
from ...utils import get_date_in_file_name
from ...utils import get_pre_trading_date
from ..backtest.order import Order


class BaseExecutor:
    """
    # Strategy framework document

    class Executor(BaseExecutor):
    """

    def execute(self, trade_account, order_list, trade_date):
        """
        return the executed result (trade_info) after trading at trade_date.
        NOTICE: trade_account will not be modified after executing.
            Parameter
            ---------
                trade_account : Account()
                order_list : list
                    [Order()]
                trade_date : pd.Timestamp
            Return
            ---------
            trade_info : list
                    [Order(), float, float, float]
        """
        raise NotImplementedError("get_execute_result for this model is not implemented.")

    def save_executed_file_from_trade_info(self, trade_info, user_path, trade_date):
        """
        Save the trade_info to the .csv transaction file in disk
        the columns of result file is
        ['date', 'stock_id', 'direction', 'trade_val', 'trade_cost', 'trade_price', 'factor']
            Parameter
            ---------
                trade_info : list of [Order(), float, float, float]
                    (order, trade_val, trade_cost, trade_price), trade_info with out factor
                user_path: str / pathlib.Path()
                    the sub folder to save user data

                transaction_path : string / pathlib.Path()
        """
        YYYY, MM, DD = str(trade_date.date()).split("-")
        folder_path = pathlib.Path(user_path) / "trade" / YYYY / MM
        if not folder_path.exists():
            folder_path.mkdir(parents=True)
        transaction_path = folder_path / "transaction_{}.csv".format(str(trade_date.date()))
        columns = [
            "date",
            "stock_id",
            "direction",
            "amount",
            "trade_val",
            "trade_cost",
            "trade_price",
            "factor",
        ]
        data = []
        for [order, trade_val, trade_cost, trade_price] in trade_info:
            data.append(
                [
                    trade_date,
                    order.stock_id,
                    order.direction,
                    order.amount,
                    trade_val,
                    trade_cost,
                    trade_price,
                    order.factor,
                ]
            )
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(transaction_path, index=False)

    def load_trade_info_from_executed_file(self, user_path, trade_date):
        YYYY, MM, DD = str(trade_date.date()).split("-")
        file_path = pathlib.Path(user_path) / "trade" / YYYY / MM / "transaction_{}.csv".format(str(trade_date.date()))
        if not file_path.exists():
            raise ValueError("File {} not exists!".format(file_path))

        filedate = get_date_in_file_name(file_path)
        transaction = pd.read_csv(file_path)
        trade_info = []
        for i in range(len(transaction)):
            date = transaction.loc[i]["date"]
            if not date == filedate:
                continue
                # raise ValueError("date in transaction file {} not equal to it's file date{}".format(date, filedate))
            order = Order(
                stock_id=transaction.loc[i]["stock_id"],
                amount=transaction.loc[i]["amount"],
                trade_date=transaction.loc[i]["date"],
                direction=transaction.loc[i]["direction"],
                factor=transaction.loc[i]["factor"],
            )
            trade_val = transaction.loc[i]["trade_val"]
            trade_cost = transaction.loc[i]["trade_cost"]
            trade_price = transaction.loc[i]["trade_price"]
            trade_info.append([order, trade_val, trade_cost, trade_price])
        return trade_info


class SimulatorExecutor(BaseExecutor):
    def __init__(self, trade_exchange, verbose=False):
        self.trade_exchange = trade_exchange
        self.verbose = verbose
        self.order_list = []

    def execute(self, trade_account, order_list, trade_date):
        """
        execute the order list, do the trading wil exchange at date.
        Will not modify the trade_account.
            Parameter
                trade_account : Account()
                order_list : list
                    list or orders
                trade_date : pd.Timestamp
            :return:
                trade_info : list of [Order(), float, float, float]
                    (order, trade_val, trade_cost, trade_price), trade_info with out factor
        """
        account = copy.deepcopy(trade_account)
        trade_info = []

        for order in order_list:
            # check holding thresh is done in strategy
            # if order.direction==0: # sell order
            #     # checking holding thresh limit for sell order
            #     if trade_account.current.get_stock_count(order.stock_id) < thresh:
            #         # can not sell this code
            #         continue
            # is order executable
            # check order
            if self.trade_exchange.check_order(order) is True:
                # execute the order
                trade_val, trade_cost, trade_price = self.trade_exchange.deal_order(order, trade_account=account)
                trade_info.append([order, trade_val, trade_cost, trade_price])
                if self.verbose:
                    if order.direction == Order.SELL:  # sell
                        print(
                            "[I {:%Y-%m-%d}]: sell {}, price {:.2f}, amount {}, value {:.2f}.".format(
                                trade_date,
                                order.stock_id,
                                trade_price,
                                order.deal_amount,
                                trade_val,
                            )
                        )
                    else:
                        print(
                            "[I {:%Y-%m-%d}]: buy {}, price {:.2f}, amount {}, value {:.2f}.".format(
                                trade_date,
                                order.stock_id,
                                trade_price,
                                order.deal_amount,
                                trade_val,
                            )
                        )

            else:
                if self.verbose:
                    print("[W {:%Y-%m-%d}]: {} wrong.".format(trade_date, order.stock_id))
                # do nothing
                pass
        return trade_info


def save_score_series(score_series, user_path, trade_date):
    """Save the score_series into a .csv file.
    The columns of saved file is
        [stock_id, score]

    Parameter
    ---------
        order_list: [Order()]
            list of Order()
        date: pd.Timestamp
            the date to save the order list
        user_path: str / pathlib.Path()
            the sub folder to save user data
    """
    user_path = pathlib.Path(user_path)
    YYYY, MM, DD = str(trade_date.date()).split("-")
    folder_path = user_path / "score" / YYYY / MM
    if not folder_path.exists():
        folder_path.mkdir(parents=True)
    file_path = folder_path / "score_{}.csv".format(str(trade_date.date()))
    score_series.to_csv(file_path)


def load_score_series(user_path, trade_date):
    """Save the score_series into a .csv file.
    The columns of saved file is
        [stock_id, score]

    Parameter
    ---------
        order_list: [Order()]
            list of Order()
        date: pd.Timestamp
            the date to save the order list
        user_path: str / pathlib.Path()
            the sub folder to save user data
    """
    user_path = pathlib.Path(user_path)
    YYYY, MM, DD = str(trade_date.date()).split("-")
    folder_path = user_path / "score" / YYYY / MM
    if not folder_path.exists():
        folder_path.mkdir(parents=True)
    file_path = folder_path / "score_{}.csv".format(str(trade_date.date()))
    score_series = pd.read_csv(file_path, index_col=0, header=None, names=["instrument", "score"])
    return score_series


def save_order_list(order_list, user_path, trade_date):
    """
    Save the order list into a json file.
    Will calculate the real amount in order according to factors at date.

    The format in json file like
    {"sell": {"stock_id": amount, ...}
    ,"buy": {"stock_id": amount, ...}}

        :param
            order_list: [Order()]
                list of Order()
            date: pd.Timestamp
                the date to save the order list
            user_path: str / pathlib.Path()
                the sub folder to save user data
    """
    user_path = pathlib.Path(user_path)
    YYYY, MM, DD = str(trade_date.date()).split("-")
    folder_path = user_path / "trade" / YYYY / MM
    if not folder_path.exists():
        folder_path.mkdir(parents=True)
    sell = {}
    buy = {}
    for order in order_list:
        if order.direction == 0:  # sell
            sell[order.stock_id] = [order.amount, order.factor]
        else:
            buy[order.stock_id] = [order.amount, order.factor]
    order_dict = {"sell": sell, "buy": buy}
    file_path = folder_path / "orderlist_{}.json".format(str(trade_date.date()))
    with file_path.open("w") as fp:
        json.dump(order_dict, fp)


def load_order_list(user_path, trade_date):
    user_path = pathlib.Path(user_path)
    YYYY, MM, DD = str(trade_date.date()).split("-")
    path = user_path / "trade" / YYYY / MM / "orderlist_{}.json".format(str(trade_date.date()))
    if not path.exists():
        raise ValueError("File {} not exists!".format(path))
    # get orders
    with path.open("r") as fp:
        order_dict = json.load(fp)
    order_list = []
    for stock_id in order_dict["sell"]:
        amount, factor = order_dict["sell"][stock_id]
        order = Order(
            stock_id=stock_id,
            amount=amount,
            trade_date=pd.Timestamp(trade_date),
            direction=Order.SELL,
            factor=factor,
        )
        order_list.append(order)
    for stock_id in order_dict["buy"]:
        amount, factor = order_dict["buy"][stock_id]
        order = Order(
            stock_id=stock_id,
            amount=amount,
            trade_date=pd.Timestamp(trade_date),
            direction=Order.BUY,
            factor=factor,
        )
        order_list.append(order)
    return order_list
