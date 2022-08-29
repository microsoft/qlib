# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import cast

import cachetools
import pandas as pd

from qlib.backtest import Exchange, Order
from qlib.backtest.decision import TradeRange, TradeRangeByTime
from qlib.constant import ONE_DAY, EPS_T
from qlib.rl.order_execution.utils import get_ticks_slice
from qlib.utils.index_data import IndexData
from .pickle_styled import BaseIntradayBacktestData


class IntradayBacktestData(BaseIntradayBacktestData):
    """Backtest data for Qlib simulator"""

    def __init__(
        self,
        order: Order,
        exchange: Exchange,
        ticks_index: pd.DatetimeIndex,
        ticks_for_order: pd.DatetimeIndex,
    ) -> None:
        self._order = order
        self._exchange = exchange
        self._start_time = ticks_for_order[0]
        self._end_time = ticks_for_order[-1]
        self.ticks_index = ticks_index
        self.ticks_for_order = ticks_for_order

        self._deal_price = cast(
            pd.Series,
            self._exchange.get_deal_price(
                self._order.stock_id,
                self._start_time,
                self._end_time,
                direction=self._order.direction,
                method=None,
            ),
        )
        self._volume = cast(
            pd.Series,
            self._exchange.get_volume(
                self._order.stock_id,
                self._start_time,
                self._end_time,
                method=None,
            ),
        )

    def __repr__(self) -> str:
        return (
            f"Order: {self._order}, Exchange: {self._exchange}, "
            f"Start time: {self._start_time}, End time: {self._end_time}"
        )

    def __len__(self) -> int:
        return len(self._deal_price)

    def get_deal_price(self) -> pd.Series:
        return self._deal_price

    def get_volume(self) -> pd.Series:
        return self._volume

    def get_time_index(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex([e[1] for e in list(self._exchange.quote_df.index)])


@cachetools.cached(  # type: ignore
    cache=cachetools.LRUCache(100),
    key=lambda order, _, __: order.key_by_day,
)
def load_qlib_backtest_data(
    order: Order,
    trade_exchange: Exchange,
    trade_range: TradeRange,
) -> IntradayBacktestData:
    data = cast(
        IndexData,
        trade_exchange.get_deal_price(
            stock_id=order.stock_id,
            start_time=order.date,
            end_time=order.date + ONE_DAY - EPS_T,
            direction=order.direction,
            method=None,
        ),
    )

    ticks_index = pd.DatetimeIndex(data.index)
    if isinstance(trade_range, TradeRangeByTime):
        ticks_for_order = get_ticks_slice(
            ticks_index,
            trade_range.start_time,
            trade_range.end_time,
            include_end=True,
        )
    else:
        ticks_for_order = None  # FIXME: implement this logic

    backtest_data = IntradayBacktestData(
        order=order,
        exchange=trade_exchange,
        ticks_index=ticks_index,
        ticks_for_order=ticks_for_order,
    )
    return backtest_data
