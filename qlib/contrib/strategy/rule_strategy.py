import copy
import warnings
import numpy as np
import pandas as pd

from ...utils import sample_feature
from ...data.data import D
from ...data.dataset.utils import get_level_index
from ...strategy.base import RuleStrategy, TradingEnhancement
from ..backtest.order import Order


class TWAPStrategy(RuleStrategy, TradingEnhancement):

    def reset(self, trade_order_list=None, **kwargs):
        super(TWAPStrategy, self).reset(**kwargs)
        TradingEnhancement.reset(self, trade_order_list=trade_order_list)
        if trade_order_list:
            self.trade_amount = {}
            for order in self.trade_order_list:
                self.trade_amount[(order.stock_id, order.direction)] = order.amount // self.trade_len
        

    def generate_order_list(self, **kwargs):
        super(TopkDropoutStrategy, self).step()
        trade_start_time, trade_end_time = self._get_calendar_time(self.trade_index)
        order_list = []
        for order in self.trade_order_list:
            _order = Order(
                stock_id=order.stock_id,
                amount=self.trade_amount[(order.stock_id, order.direction)],
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=order.direction,  # 1 for buy
                factor=order.factor,
            )
            order_list.append(_order)
        return order_list

class SBBStrategyBase(RuleStrategy, TradingEnhancement):
    """
        (S)elect the (B)etter one among every two adjacent trading (B)ars to sell or buy.
    """
    TREND_MID = 0
    TREND_SHORT = 1
    TREND_LONG = 2

    def reset(self, trade_order_list=None, **kwargs):
        super(SBBStrategyBase, self).reset(**kwargs)
        TradingEnhancement.reset(self, trade_order_list=trade_order_list)
        if trade_order_list:
            self.trade_amount = {}
            self.trade_trend = {} 
            for order in self.trade_order_list:
                self.trade_amount[(order.stock_id, order.direction)] = order.amount // self.trade_len
                self.trade_trend[(order.stock_id, order.direction)] = self.TREND_MID
        

    def _pred_price_trend(self, stock_id, pred_start_time=None, pred_end_time=None):
        raise NotImplementedError("pred_price_trend method is not implemented!")

    def generate_order_list(self, **kwargs):
        super(SBBStrategyBase, self).step()
        trade_start_time, trade_end_time = self._get_calendar_time(self.trade_index)
        pred_start_time, pred_end_time = self._get_calendar_time(self.trade_index, shift=1)
        order_list = []
        for order in self.trade_order_list:
            if self.trade_index % 2 == 1:
                _pred_trend = self._pred_price_trend(order.stock_id)
            else:
                _pred_trend = self.trade_trend[(order.stock_id, order.direction)]
            if _pred_trend == self.TREND_MID:
                _order = Order(
                    stock_id=order.stock_id,
                    amount=self.trade_amount[(order.stock_id, order.direction)],
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                    direction=order.direction,  # 1 for buy
                    factor=order.factor,
                )
                order_list.append(_order)
            else:
                if self.trade_index % 2 == 1:
                    if _pred_trend == self.TREND_SHORT and order.direction == order.SELL or _pred_trend == self.TREND_LONG and order.direction == order.BUY:
                        _order = Order(
                            stock_id=order.stock_id,
                            amount=2*self.trade_amount[(order.stock_id, order.direction)],
                            start_time=trade_start_time,
                            end_time=trade_end_time,
                            direction=order.direction,  # 1 for buy
                            factor=order.factor,
                        )
                        order_list.append(_order)
                else:
                    if _pred_trend == self.TREND_SHORT and order.direction == order.BUY or _pred_trend == self.TREND_LONG and order.direction == order.SELL:
                        _order = Order(
                            stock_id=order.stock_id,
                            amount=2*self.trade_amount[(order.stock_id, order.direction)],
                            start_time=trade_start_time,
                            end_time=trade_end_time,
                            direction=order.direction,  # 1 for buy
                            factor=order.factor,
                        )       
                        order_list.append(_order)
            if self.trade_index % 2 == 1:  
                self.trade_trend[(order.stock_id, order.direction)] = _pred_trend

        return order_list

    
class SBBStrategyEMA(SBBStrategyBase):
    """
        (S)elect the (B)etter one among every two adjacent trading (B)ars to sell or buy with (EMA).
    """
    def __init__(
        self, 
        step_bar, 
        start_time=None, 
        end_time=None, 
        instruments="csi300",
        freq="day",
        **kwargs,
    ):
        super(SBBStrategyEMA, self).__init__(step_bar, start_time, end_time, **kwargs)
        if instruments is None:
            warnings.warn("`instruments` is not set, will load all stocks")
            self.instruments = "all"
        if isinstance(instruments, str):
            self.instruments = D.instruments(instruments)
        self.freq = freq

    def _convert_index_format(self, df):
        if get_level_index(df, level="datetime") == 1:
            df = df.swaplevel().sort_index()
        return df

    def _reset_trade_calendar(self, start_time=None, end_time=None):
        super(SBBStrategyEMA, self)._reset_trade_calendar(start_time=start_time, end_time=end_time)
        if self.start_time and self.end_time:
            fields = ["EMA($close, 10)-EMA($close, 20)"]
            signal_start_time, _ = self._get_calendar_time(trade_index=self.trade_index, shift=1)
            signal_df = D.features(self.instruments, fields, start_time=signal_start_time, end_time=self.end_time, freq=self.freq)
            signal_df = self._convert_index_format(signal_df)
            signal_df.columns = ["signal"]
            self.signal = {}
            for stock_id, stock_val in signal_df.groupby(level="instrument"):
                self.signal[stock_id] = stock_val
            
    def _pred_price_trend(self, stock_id, pred_start_time=None, pred_end_time=None):
        if stock_id not in self.signal:
            return self.TREND_MID
        else:
            _sample_signal = sample_feature(self.signal[stock_id], pred_start_time, pred_end_time, fields="signal", method="last")
            if _sample_signal is None or _sample_signal.iloc[0] == 0:
                return self.TREND_MID
            elif _sample_signal.iloc[0] > 0:
                return self.TREND_LONG
            else:
                return self.TREND_SHORT
            