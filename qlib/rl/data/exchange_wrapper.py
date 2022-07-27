from typing import cast

import pandas as pd

from qlib.backtest import Exchange, Order
from qlib.rl.data.pickle_styled import IntradayBacktestData


class QlibIntradayBacktestData(IntradayBacktestData):
    """Backtest data for Qlib simulator"""

    def __init__(self, order: Order, exchange: Exchange, start_time: pd.Timestamp, end_time: pd.Timestamp) -> None:
        super(QlibIntradayBacktestData, self).__init__()
        self._order = order
        self._exchange = exchange
        self._start_time = start_time
        self._end_time = end_time

    def __repr__(self) -> str:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def get_deal_price(self) -> pd.Series:
        return cast(
            pd.Series,
            self._exchange.get_deal_price(
                self._order.stock_id,
                self._start_time,
                self._end_time,
                direction=self._order.direction,
                method=None,
            ),
        )

    def get_volume(self) -> pd.Series:
        return cast(
            pd.Series,
            self._exchange.get_volume(
                self._order.stock_id,
                self._start_time,
                self._end_time,
                method=None,
            ),
        )

    def get_time_index(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex([e[1] for e in list(self._exchange.quote_df.index)])
