# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import pandas as pd
from dataclasses import dataclass, field
from typing import ClassVar

@dataclass
class Order:
    """
    stock_id : str
    amount : float
    start_time : pd.Timestamp
        closed start time for order generation
    end_time : pd.Timestamp
        closed end time for order generation
    direction : Order.SELL for sell; Order.BUY for buy
    factor : float
            presents the weight factor assigned in Exchange()
    """
    stock_id : str
    amount : float
    start_time : pd.Timestamp
    end_time : pd.Timestamp
    direction : int
    factor : float
    deal_amount : float = field(init=False)
    SELL : ClassVar[int] = 0
    BUY : ClassVar[int] = 1
    

    def __post_init__(self):
        if self.direction not in {Order.SELL, Order.BUY}:
            raise NotImplementedError("direction not supported, `Order.SELL` for sell, `Order.BUY` for buy")
        self.deal_amount = 0

