# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import copy
import pathlib
from typing import Dict, List
import pandas as pd
import numpy as np
from .order import Order


class BasePosition:
    """
    The Position want to maintain the position like a dictionary
    Please refer to the `Position` class for the position
    """

    def __init__(self, cash=0.0, *args, **kwargs) -> None:
        pass

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

    def get_cash(self) -> float:
        """

        Returns
        -------
        float:
            the cash in position
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
        it is meaningful in the beginning or the end of each trade date

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

    def __init__(self, cash=0, position_dict={}, now_account_value=0):
        # NOTE: The position dict must be copied!!!
        # Otherwise the initial value
        self.init_cash = cash
        self.position = position_dict.copy()
        self.position["cash"] = cash
        self.position["now_account_value"] = now_account_value

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
            # decrease the amount of stock
            self.position[stock_id]["amount"] -= trade_amount
            # check if to delete
            if self.position[stock_id]["amount"] < -1e-5:
                raise ValueError(
                    "only have {} {}, require {}".format(self.position[stock_id]["amount"], stock_id, trade_amount)
                )
            elif abs(self.position[stock_id]["amount"]) <= 1e-5:
                self._del_stock(stock_id)

        self.position["cash"] += trade_val - cost

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

    def update_cash(self, cash):
        self.position["cash"] = cash

    def calculate_stock_value(self):
        stock_list = self.get_stock_list()
        value = 0
        for stock_id in stock_list:
            value += self.position[stock_id]["amount"] * self.position[stock_id]["price"]
        return value

    def calculate_value(self):
        value = self.calculate_stock_value()
        value += self.position["cash"]
        return value

    def get_stock_list(self):
        stock_list = list(set(self.position.keys()) - {"cash", "now_account_value"})
        return stock_list

    def get_stock_price(self, code):
        return self.position[code]["price"]

    def get_stock_amount(self, code):
        return self.position[code]["amount"]

    def get_stock_count(self, code, bar):
        """the days the account has been hold, it may be used in some special strategies"""
        if f"count_{bar}" in self.position[code]:
            return self.position[code][f"count_{bar}"]
        else:
            return 0

    def get_stock_weight(self, code):
        return self.position[code]["weight"]

    def get_cash(self):
        return self.position["cash"]

    def get_stock_amount_dict(self):
        """generate stock amount dict {stock_id : amount of stock}"""
        d = {}
        stock_list = self.get_stock_list()
        for stock_code in stock_list:
            d[stock_code] = self.get_stock_amount(code=stock_code)
        return d

    def get_stock_weight_dict(self, only_stock=False):
        """get_stock_weight_dict
        generate stock weight fict {stock_id : value weight of stock in the position}
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

    def save_position(self, path):
        path = pathlib.Path(path)
        p = copy.deepcopy(self.position)
        cash = pd.Series(dtype=float)
        cash["init_cash"] = self.init_cash
        cash["cash"] = p["cash"]
        cash["now_account_value"] = p["now_account_value"]
        del p["cash"]
        del p["now_account_value"]
        positions = pd.DataFrame.from_dict(p, orient="index")
        with pd.ExcelWriter(path) as writer:
            positions.to_excel(writer, sheet_name="position")
            cash.to_excel(writer, sheet_name="info")

    def load_position(self, path):
        """load position information from a file
        should have format below
        sheet "position"
            columns: ['stock', f'count_{bar}', 'amount', 'price', 'weight']
                f'count_{bar}': <how many bars the security has been hold>,
                'amount': <the amount of the security>,
                'price': <the close price of security in the last trading day>,
                'weight': <the security weight of total position value>,

        sheet "cash"
            index: ['init_cash', 'cash', 'now_account_value']
            'init_cash': <inital cash when account was created>,
            'cash': <current cash in account>,
            'now_account_value': <current total account value, should equal to sum(price[stock]*amount[stock])>
        """
        path = pathlib.Path(path)
        positions = pd.read_excel(open(path, "rb"), sheet_name="position", index_col=0)
        cash_record = pd.read_excel(open(path, "rb"), sheet_name="info", index_col=0)
        positions = positions.to_dict(orient="index")
        init_cash = cash_record.loc["init_cash"].values[0]
        cash = cash_record.loc["cash"].values[0]
        now_account_value = cash_record.loc["now_account_value"].values[0]
        # assign values
        self.position = {}
        self.init_cash = init_cash
        self.position = positions
        self.position["cash"] = cash
        self.position["now_account_value"] = now_account_value


class InfPosition(BasePosition):
    """
    Position with infinite cash and amount.

    This is useful for generating random orders.
    """

    def skip_update(self) -> bool:
        """ Updating state is meaningless for InfPosition """
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

    def get_cash(self) -> float:
        return np.inf

    def get_stock_amount_dict(self) -> Dict:
        raise NotImplementedError(f"InfPosition doesn't support get_stock_amount_dict")

    def get_stock_weight_dict(self, only_stock: bool) -> Dict:
        raise NotImplementedError(f"InfPosition doesn't support get_stock_weight_dict")

    def add_count_all(self, bar):
        raise NotImplementedError(f"InfPosition doesn't support get_stock_weight_dict")

    def update_weight_all(self):
        raise NotImplementedError(f"InfPosition doesn't support update_weight_all")
