# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


class Order:

    SELL = 0
    BUY = 1

    def __init__(self, stock_id, amount, trade_date, direction, factor):
        """Parameter
        direction : Order.SELL for sell; Order.BUY for buy
        stock_id : str
        amount : float
        trade_date : pd.Timestamp
        factor : float
            presents the weight factor assigned in Exchange()
        """
        # check direction
        if direction not in {Order.SELL, Order.BUY}:
            raise NotImplementedError("direction not supported, `Order.SELL` for sell, `Order.BUY` for buy")
        self.stock_id = stock_id
        # amount of generated orders
        self.amount = amount
        # amount of successfully completed orders
        self.deal_amount = 0
        self.trade_date = trade_date
        self.direction = direction
        self.factor = factor
