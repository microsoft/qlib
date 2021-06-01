# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import copy
import pathlib
import pandas as pd
import numpy as np
from .order import Order

"""
Position module
"""

"""
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


class Position:
    """Position"""

    def __init__(self, cash=0, position_dict={}, today_account_value=0):
        # NOTE: The position dict must be copied!!!
        # Otherwise the initial value
        self.init_cash = cash
        self.position = position_dict.copy()
        self.position["cash"] = cash
        self.position["today_account_value"] = today_account_value

    def init_stock(self, stock_id, amount, price=None):
        self.position[stock_id] = {}
        self.position[stock_id]["count"] = 0  # update count in the end of this date
        self.position[stock_id]["amount"] = amount
        self.position[stock_id]["price"] = price
        self.position[stock_id]["weight"] = 0  # update the weight in the end of the trade date

    def buy_stock(self, stock_id, trade_val, cost, trade_price):
        trade_amount = trade_val / trade_price
        if stock_id not in self.position:
            self.init_stock(stock_id=stock_id, amount=trade_amount, price=trade_price)
        else:
            # exist, add amount
            self.position[stock_id]["amount"] += trade_amount

        self.position["cash"] -= trade_val + cost

    def sell_stock(self, stock_id, trade_val, cost, trade_price):
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
                self.del_stock(stock_id)

        self.position["cash"] += trade_val - cost

    def del_stock(self, stock_id):
        del self.position[stock_id]

    def update_order(self, order, trade_val, cost, trade_price):
        # handle order, order is a order class, defined in exchange.py
        if order.direction == Order.BUY:
            # BUY
            self.buy_stock(order.stock_id, trade_val, cost, trade_price)
        elif order.direction == Order.SELL:
            # SELL
            self.sell_stock(order.stock_id, trade_val, cost, trade_price)
        else:
            raise NotImplementedError("do not suppotr order direction {}".format(order.direction))

    def update_stock_price(self, stock_id, price):
        self.position[stock_id]["price"] = price

    def update_stock_count(self, stock_id, count):
        self.position[stock_id]["count"] = count

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
        stock_list = list(set(self.position.keys()) - {"cash", "today_account_value"})
        return stock_list

    def get_stock_price(self, code):
        return self.position[code]["price"]

    def get_stock_amount(self, code):
        return self.position[code]["amount"]

    def get_stock_count(self, code):
        return self.position[code]["count"]

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

    def add_count_all(self):
        stock_list = self.get_stock_list()
        for code in stock_list:
            self.position[code]["count"] += 1

    def update_weight_all(self):
        weight_dict = self.get_stock_weight_dict()
        for stock_code, weight in weight_dict.items():
            self.update_stock_weight(stock_code, weight)

    def save_position(self, path, last_trade_date):
        path = pathlib.Path(path)
        p = copy.deepcopy(self.position)
        cash = pd.Series(dtype=float)
        cash["init_cash"] = self.init_cash
        cash["cash"] = p["cash"]
        cash["today_account_value"] = p["today_account_value"]
        cash["last_trade_date"] = str(last_trade_date.date()) if last_trade_date else None
        del p["cash"]
        del p["today_account_value"]
        positions = pd.DataFrame.from_dict(p, orient="index")
        with pd.ExcelWriter(path) as writer:
            positions.to_excel(writer, sheet_name="position")
            cash.to_excel(writer, sheet_name="info")

    def load_position(self, path):
        """load position information from a file
        should have format below
        sheet "position"
            columns: ['stock', 'count', 'amount', 'price', 'weight']
                'count': <how many days the security has been hold>,
                'amount': <the amount of the security>,
                'price': <the close price of security in the last trading day>,
                'weight': <the security weight of total position value>,

        sheet "cash"
            index: ['init_cash', 'cash', 'today_account_value']
            'init_cash': <inital cash when account was created>,
            'cash': <current cash in account>,
            'today_account_value': <current total account value, should equal to sum(price[stock]*amount[stock])>
        """
        path = pathlib.Path(path)
        positions = pd.read_excel(open(path, "rb"), sheet_name="position", index_col=0)
        cash_record = pd.read_excel(open(path, "rb"), sheet_name="info", index_col=0)
        positions = positions.to_dict(orient="index")
        init_cash = cash_record.loc["init_cash"].values[0]
        cash = cash_record.loc["cash"].values[0]
        today_account_value = cash_record.loc["today_account_value"].values[0]
        last_trade_date = cash_record.loc["last_trade_date"].values[0]

        # assign values
        self.position = {}
        self.init_cash = init_cash
        self.position = positions
        self.position["cash"] = cash
        self.position["today_account_value"] = today_account_value

        return None if pd.isna(last_trade_date) else pd.Timestamp(last_trade_date)
