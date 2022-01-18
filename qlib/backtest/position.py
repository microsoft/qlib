# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import copy
import pathlib
from typing import Dict, List, Union

import pandas as pd
from datetime import timedelta
import numpy as np

from .decision import Order
from ..data.data import D


class BasePosition:
    """
    The Position want to maintain the position like a dictionary
    Please refer to the `Position` class for the position
    """

    def __init__(self, *args, cash=0.0, **kwargs):
        self._settle_type = self.ST_NO

    def skip_update(self) -> bool:
        """
        Should we skip updating operation for this position
        For example, updating is meaningless for InfPosition

        Returns
        -------
        bool:
            should we skip the updating operator
        """
        return False

    def check_stock(self, stock_id: str) -> bool:
        """
        check if is the stock in the position

        Parameters
        ----------
        stock_id : str
            the id of the stock

        Returns
        -------
        bool:
            if is the stock in the position
        """
        raise NotImplementedError(f"Please implement the `check_stock` method")

    def update_order(self, order: Order, trade_val: float, cost: float, trade_price: float):
        """
        Parameters
        ----------
        order : Order
            the order to update the position
        trade_val : float
            the trade value(money) of dealing results
        cost : float
            the trade cost of the dealing results
        trade_price : float
            the trade price of the dealing results
        """
        raise NotImplementedError(f"Please implement the `update_order` method")

    def update_stock_price(self, stock_id, price: float):
        """
        Updating the latest price of the order
        The useful when clearing balance at each bar end

        Parameters
        ----------
        stock_id :
            the id of the stock
        price : float
            the price to be updated
        """
        raise NotImplementedError(f"Please implement the `update stock price` method")

    def calculate_stock_value(self) -> float:
        """
        calculate the value of the all assets except cash in the position

        Returns
        -------
        float:
            the value(money) of all the stock
        """
        raise NotImplementedError(f"Please implement the `calculate_stock_value` method")

    def get_stock_list(self) -> List:
        """
        Get the list of stocks in the position.
        """
        raise NotImplementedError(f"Please implement the `get_stock_list` method")

    def get_stock_price(self, code) -> float:
        """
        get the latest price of the stock

        Parameters
        ----------
        code :
            the code of the stock
        """
        raise NotImplementedError(f"Please implement the `get_stock_price` method")

    def get_stock_amount(self, code) -> float:
        """
        get the amount of the stock

        Parameters
        ----------
        code :
            the code of the stock

        Returns
        -------
        float:
            the amount of the stock
        """
        raise NotImplementedError(f"Please implement the `get_stock_amount` method")

    def get_cash(self, include_settle: bool = False) -> float:
        """

        Returns
        -------
        float:
            the available(tradable) cash in position
        include_settle:
            will the unsettled(delayed) cash included
            Default: not include those unavailable cash
        """
        raise NotImplementedError(f"Please implement the `get_cash` method")

    def get_stock_amount_dict(self) -> Dict:
        """
        generate stock amount dict {stock_id : amount of stock}

        Returns
        -------
        Dict:
            {stock_id : amount of stock}
        """
        raise NotImplementedError(f"Please implement the `get_stock_amount_dict` method")

    def get_stock_weight_dict(self, only_stock: bool = False) -> Dict:
        """
        generate stock weight dict {stock_id : value weight of stock in the position}
        it is meaningful in the beginning or the end of each trade step
        - During execution of each trading step, the weight may be not consistent with the portfolio value

        Parameters
        ----------
        only_stock : bool
            If only_stock=True, the weight of each stock in total stock will be returned
            If only_stock=False, the weight of each stock in total assets(stock + cash) will be returned

        Returns
        -------
        Dict:
            {stock_id : value weight of stock in the position}
        """
        raise NotImplementedError(f"Please implement the `get_stock_weight_dict` method")

    def add_count_all(self, bar):
        """
        Will be called at the end of each bar on each level

        Parameters
        ----------
        bar :
            The level to be updated
        """
        raise NotImplementedError(f"Please implement the `add_count_all` method")

    def update_weight_all(self):
        """
        Updating the position weight;

        # TODO: this function is a little weird. The weight data in the position is in a wrong state after dealing order
        # and before updating weight.

        Parameters
        ----------
        bar :
            The level to be updated
        """
        raise NotImplementedError(f"Please implement the `add_count_all` method")

    ST_CASH = "cash"
    ST_NO = None

    def settle_start(self, settle_type: str):
        """
        settlement start
        It will act like start and commit a transaction

        Parameters
        ----------
        settle_type : str
            Should we make delay the settlement in each execution (each execution will make the executor a step forward)
            - "cash": make the cash settlement delayed.
                - The cash you get can't be used in current step (e.g. you can't sell a stock to get cash to buy another
                        stock)
            - None: not settlement mechanism
            - TODO: other assets will be supported in the future.
        """
        raise NotImplementedError(f"Please implement the `settle_conf` method")

    def settle_commit(self):
        """
        settlement commit

        Parameters
        ----------
        settle_type : str
            please refer to the documents of Executor
        """
        raise NotImplementedError(f"Please implement the `settle_commit` method")

    def __str__(self):
        return self.__dict__.__str__()

    def __repr__(self):
        return self.__dict__.__repr__()


class Position(BasePosition):
    """Position

    current state of position
    a typical example is :{
      <instrument_id>: {
        'count': <how many days the security has been hold>,
        'amount': <the amount of the security>,
        'price': <the close price of security in the last trading day>,
        'weight': <the security weight of total position value>,
      },
    }
    """

    def __init__(self, cash: float = 0, position_dict: Dict[str, Dict[str, float]] = {}):
        """Init position by cash and position_dict.

        Parameters
        ----------
        start_time :
            the start time of backtest. It's for filling the initial value of stocks.
        cash : float, optional
            initial cash in account, by default 0
        position_dict : Dict[
                            stock_id,
                            Union[
                                int,  # it is equal to {"amount": int}
                                {"amount": int, "price"(optional): float},
                            ]
                        ]
            initial stocks with parameters amount and price,
            if there is no price key in the dict of stocks, it will be filled by _fill_stock_value.
            by default {}.
        """
        super().__init__()

        # NOTE: The position dict must be copied!!!
        # Otherwise the initial value
        self.init_cash = cash
        self.position = position_dict.copy()
        for stock in self.position:
            if isinstance(self.position[stock], int):
                self.position[stock] = {"amount": self.position[stock]}
        self.position["cash"] = cash

        # If the stock price information is missing, the account value will not be calculated temporarily
        try:
            self.position["now_account_value"] = self.calculate_value()
        except KeyError:
            pass

    def fill_stock_value(self, start_time: Union[str, pd.Timestamp], freq: str, last_days: int = 30):
        """fill the stock value by the close price of latest last_days from qlib.

        Parameters
        ----------
        start_time :
            the start time of backtest.
        last_days : int, optional
            the days to get the latest close price, by default 30.
        """
        stock_list = []
        for stock in self.position:
            if not isinstance(self.position[stock], dict):
                continue
            if ("price" not in self.position[stock]) or (self.position[stock]["price"] is None):
                stock_list.append(stock)

        if len(stock_list) == 0:
            return

        start_time = pd.Timestamp(start_time)
        # note that start time is 2020-01-01 00:00:00 if raw start time is "2020-01-01"
        price_end_time = start_time
        price_start_time = start_time - timedelta(days=last_days)
        price_df = D.features(
            stock_list, ["$close"], price_start_time, price_end_time, freq=freq, disk_cache=True
        ).dropna()
        price_dict = price_df.groupby(["instrument"]).tail(1).reset_index(level=1, drop=True)["$close"].to_dict()

        if len(price_dict) < len(stock_list):
            lack_stock = set(stock_list) - set(price_dict)
            raise ValueError(f"{lack_stock} doesn't have close price in qlib in the latest {last_days} days")

        for stock in stock_list:
            self.position[stock]["price"] = price_dict[stock]
        self.position["now_account_value"] = self.calculate_value()

    def _init_stock(self, stock_id, amount, price=None):
        """
        initialization the stock in current position

        Parameters
        ----------
        stock_id :
            the id of the stock
        amount : float
            the amount of the stock
        price :
             the price when buying the init stock
        """
        self.position[stock_id] = {}
        self.position[stock_id]["amount"] = amount
        self.position[stock_id]["price"] = price
        self.position[stock_id]["weight"] = 0  # update the weight in the end of the trade date

    def _buy_stock(self, stock_id, trade_val, cost, trade_price):
        trade_amount = trade_val / trade_price
        if stock_id not in self.position:
            self._init_stock(stock_id=stock_id, amount=trade_amount, price=trade_price)
        else:
            # exist, add amount
            self.position[stock_id]["amount"] += trade_amount

        self.position["cash"] -= trade_val + cost

    def _sell_stock(self, stock_id, trade_val, cost, trade_price):
        trade_amount = trade_val / trade_price
        if stock_id not in self.position:
            raise KeyError("{} not in current position".format(stock_id))
        else:
            if np.isclose(self.position[stock_id]["amount"], trade_amount):
                # Selling all the stocks
                # we use np.isclose instead of abs(<the final amount>) <= 1e-5  because `np.isclose` consider both ralative amount and absolute amount
                # Using  abs(<the final amount>) <= 1e-5 will result in error when the amount is large
                self._del_stock(stock_id)
            else:
                # decrease the amount of stock
                self.position[stock_id]["amount"] -= trade_amount
                # check if to delete
                if self.position[stock_id]["amount"] < -1e-5:
                    raise ValueError(
                        "only have {} {}, require {}".format(self.position[stock_id]["amount"], stock_id, trade_amount)
                    )

        new_cash = trade_val - cost
        if self._settle_type == self.ST_CASH:
            self.position["cash_delay"] += new_cash
        elif self._settle_type == self.ST_NO:
            self.position["cash"] += new_cash
        else:
            raise NotImplementedError(f"This type of input is not supported")

    def _del_stock(self, stock_id):
        del self.position[stock_id]

    def check_stock(self, stock_id):
        return stock_id in self.position

    def update_order(self, order, trade_val, cost, trade_price):
        # handle order, order is a order class, defined in exchange.py
        if order.direction == Order.BUY:
            # BUY
            self._buy_stock(order.stock_id, trade_val, cost, trade_price)
        elif order.direction == Order.SELL:
            # SELL
            self._sell_stock(order.stock_id, trade_val, cost, trade_price)
        else:
            raise NotImplementedError("do not support order direction {}".format(order.direction))

    def update_stock_price(self, stock_id, price):
        self.position[stock_id]["price"] = price

    def update_stock_count(self, stock_id, bar, count):
        self.position[stock_id][f"count_{bar}"] = count

    def update_stock_weight(self, stock_id, weight):
        self.position[stock_id]["weight"] = weight

    def calculate_stock_value(self):
        stock_list = self.get_stock_list()
        value = 0
        for stock_id in stock_list:
            value += self.position[stock_id]["amount"] * self.position[stock_id]["price"]
        return value

    def calculate_value(self):
        value = self.calculate_stock_value()
        value += self.position["cash"] + self.position.get("cash_delay", 0.0)
        return value

    def get_stock_list(self):
        stock_list = list(set(self.position.keys()) - {"cash", "now_account_value", "cash_delay"})
        return stock_list

    def get_stock_price(self, code):
        return self.position[code]["price"]

    def get_stock_amount(self, code):
        return self.position[code]["amount"] if code in self.position else 0

    def get_stock_count(self, code, bar):
        """the days the account has been hold, it may be used in some special strategies"""
        if f"count_{bar}" in self.position[code]:
            return self.position[code][f"count_{bar}"]
        else:
            return 0

    def get_stock_weight(self, code):
        return self.position[code]["weight"]

    def get_cash(self, include_settle=False):
        cash = self.position["cash"]
        if include_settle:
            cash += self.position.get("cash_delay", 0.0)
        return cash

    def get_stock_amount_dict(self):
        """generate stock amount dict {stock_id : amount of stock}"""
        d = {}
        stock_list = self.get_stock_list()
        for stock_code in stock_list:
            d[stock_code] = self.get_stock_amount(code=stock_code)
        return d

    def get_stock_weight_dict(self, only_stock=False):
        """get_stock_weight_dict
        generate stock weight dict {stock_id : value weight of stock in the position}
        it is meaningful in the beginning or the end of each trade date

        :param only_stock: If only_stock=True, the weight of each stock in total stock will be returned
                           If only_stock=False, the weight of each stock in total assets(stock + cash) will be returned
        """
        if only_stock:
            position_value = self.calculate_stock_value()
        else:
            position_value = self.calculate_value()
        d = {}
        stock_list = self.get_stock_list()
        for stock_code in stock_list:
            d[stock_code] = self.position[stock_code]["amount"] * self.position[stock_code]["price"] / position_value
        return d

    def add_count_all(self, bar):
        stock_list = self.get_stock_list()
        for code in stock_list:
            if f"count_{bar}" in self.position[code]:
                self.position[code][f"count_{bar}"] += 1
            else:
                self.position[code][f"count_{bar}"] = 1

    def update_weight_all(self):
        weight_dict = self.get_stock_weight_dict()
        for stock_code, weight in weight_dict.items():
            self.update_stock_weight(stock_code, weight)

    def settle_start(self, settle_type):
        assert self._settle_type == self.ST_NO, "Currently, settlement can't be nested!!!!!"
        self._settle_type = settle_type
        if settle_type == self.ST_CASH:
            self.position["cash_delay"] = 0.0

    def settle_commit(self):
        if self._settle_type != self.ST_NO:
            if self._settle_type == self.ST_CASH:
                self.position["cash"] += self.position["cash_delay"]
                del self.position["cash_delay"]
            else:
                raise NotImplementedError(f"This type of input is not supported")
            self._settle_type = self.ST_NO


class InfPosition(BasePosition):
    """
    Position with infinite cash and amount.

    This is useful for generating random orders.
    """

    def skip_update(self) -> bool:
        """Updating state is meaningless for InfPosition"""
        return True

    def check_stock(self, stock_id: str) -> bool:
        # InfPosition always have any stocks
        return True

    def update_order(self, order: Order, trade_val: float, cost: float, trade_price: float):
        pass

    def update_stock_price(self, stock_id, price: float):
        pass

    def calculate_stock_value(self) -> float:
        """
        Returns
        -------
        float:
            infinity stock value
        """
        return np.inf

    def get_stock_list(self) -> List:
        raise NotImplementedError(f"InfPosition doesn't support stock list position")

    def get_stock_price(self, code) -> float:
        """the price of the inf position is meaningless"""
        return np.nan

    def get_stock_amount(self, code) -> float:
        return np.inf

    def get_cash(self, include_settle=False) -> float:
        return np.inf

    def get_stock_amount_dict(self) -> Dict:
        raise NotImplementedError(f"InfPosition doesn't support get_stock_amount_dict")

    def get_stock_weight_dict(self, only_stock: bool) -> Dict:
        raise NotImplementedError(f"InfPosition doesn't support get_stock_weight_dict")

    def add_count_all(self, bar):
        raise NotImplementedError(f"InfPosition doesn't support add_count_all")

    def update_weight_all(self):
        raise NotImplementedError(f"InfPosition doesn't support update_weight_all")

    def settle_start(self, settle_type: str):
        pass

    def settle_commit(self):
        pass
