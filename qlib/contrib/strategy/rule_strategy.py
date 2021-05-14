import copy
import warnings
import numpy as np
import pandas as pd
from typing import Union

from ...utils.sample import sample_feature
from ...data.data import D
from ...data.dataset.utils import convert_index_format
from ...strategy.base import RuleStrategy, OrderEnhancement
from ..backtest.order import Order
from ..backtest.faculty import common_faculty


class TWAPStrategy(RuleStrategy, OrderEnhancement):
    """TWAP Strategy for trading"""

    def __init__(
        self,
        step_bar,
        start_time=None,
        end_time=None,
        trade_exchange=None,
        trade_order_list=[],
        **kwargs,
    ):
        """
        Parameters
        ----------
        trade_exchange : Exchange, optional
            exchange that provides market info, by default None
            - If `trade_exchange` is None, self.trade_exchange will be set with common_faculty
        trade_order_list : list, optional
            order list to trade, which the strategy will trade in [start_time , end_time] , by default []
        """
        super(TWAPStrategy, self).__init__(step_bar, start_time, end_time, **kwargs)
        self.trade_exchange = common_faculty.trade_exchange if trade_exchange is None else trade_exchange
        self.trade_order_list = trade_order_list

    def reset(self, trade_order_list: list = None, **kwargs):
        super(TWAPStrategy, self).reset(**kwargs)
        OrderEnhancement.reset(self, trade_order_list=trade_order_list)
        if trade_order_list is not None:
            self.trade_amount = {}
            for order in self.trade_order_list:
                self.trade_amount[(order.stock_id, order.direction)] = order.amount

    def generate_order_list(self, execute_state):
        super(TWAPStrategy, self).step()
        trade_info = execute_state.get("trade_info")
        for order, _, _, _ in trade_info:
            self.trade_amount[(order.stock_id, order.direction)] -= order.deal_amount

        trade_start_time, trade_end_time = self._get_calendar_time(self.trade_index)
        order_list = []
        for order in self.trade_order_list:
            if not self.trade_exchange.is_stock_tradable(
                stock_id=order.stock_id, start_time=trade_start_time, end_time=trade_end_time
            ):
                continue
            _amount_trade_unit = self.trade_exchange.get_amount_of_trade_unit(order.factor)
            _order_amount = None
            if _amount_trade_unit is None:
                _order_amount = self.trade_amount[(order.stock_id, order.direction)] / (
                    self.trade_len - self.trade_index + 1
                )
            if self.trade_amount[(order.stock_id, order.direction)] >= _amount_trade_unit:
                trade_unit_cnt = int(self.trade_amount[(order.stock_id, order.direction)] // _amount_trade_unit)
                _order_amount = (
                    (trade_unit_cnt + self.trade_len - self.trade_index)
                    // (self.trade_len - self.trade_index + 1)
                    * _amount_trade_unit
                )

            if order.direction == order.SELL:
                if self.trade_amount[(order.stock_id, order.direction)] > 1e-5 and (
                    _order_amount is None or self.trade_index == self.trade_len
                ):
                    _order_amount = self.trade_amount[(order.stock_id, order.direction)]

            if _order_amount:
                _order_amount = min(_order_amount, self.trade_amount[(order.stock_id, order.direction)])
                _order = Order(
                    stock_id=order.stock_id,
                    amount=_order_amount,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                    direction=order.direction,  # 1 for buy
                    factor=order.factor,
                )
                order_list.append(_order)
        return order_list


class SBBStrategyBase(RuleStrategy, OrderEnhancement):
    """
    (S)elect the (B)etter one among every two adjacent trading (B)ars to sell or buy.
    """

    TREND_MID = 0
    TREND_SHORT = 1
    TREND_LONG = 2

    def __init__(
        self,
        step_bar,
        start_time=None,
        end_time=None,
        trade_exchange=None,
        trade_order_list=[],
        **kwargs,
    ):
        """
        Parameters
        ----------
        trade_exchange : Exchange, optional
            exchange that provides market info, by default None
            - If `trade_exchange` is None, self.trade_exchange will be set with common_faculty
        trade_order_list : list, optional
            order list to trade, which the strategy will trade in [start_time , end_time] , by default []
        """
        super(SBBStrategyBase, self).__init__(step_bar, start_time, end_time, **kwargs)
        self.trade_exchange = common_faculty.trade_exchange if trade_exchange is None else trade_exchange
        self.trade_order_list = trade_order_list

    def reset(self, trade_order_list=None, **kwargs):
        super(SBBStrategyBase, self).reset(**kwargs)
        OrderEnhancement.reset(self, trade_order_list=trade_order_list)
        if trade_order_list is not None:
            self.trade_trend = {}
            self.trade_amount = {}
            for order in self.trade_order_list:
                self.trade_trend[(order.stock_id, order.direction)] = self.TREND_MID
                self.trade_amount[(order.stock_id, order.direction)] = order.amount

    def _pred_price_trend(self, stock_id, pred_start_time=None, pred_end_time=None):
        raise NotImplementedError("pred_price_trend method is not implemented!")

    def generate_order_list(self, execute_state):
        super(SBBStrategyBase, self).step()

        trade_info = execute_state.get("trade_info")
        for order, _, _, _ in trade_info:
            self.trade_amount[(order.stock_id, order.direction)] -= order.deal_amount

        trade_start_time, trade_end_time = self._get_calendar_time(self.trade_index)
        pred_start_time, pred_end_time = self._get_calendar_time(self.trade_index, shift=1)
        order_list = []
        for order in self.trade_order_list:
            if self.trade_index % 2 == 1:
                _pred_trend = self._pred_price_trend(order.stock_id, pred_start_time, pred_end_time)
            else:
                _pred_trend = self.trade_trend[(order.stock_id, order.direction)]

            if not self.trade_exchange.is_stock_tradable(
                stock_id=order.stock_id, start_time=trade_start_time, end_time=trade_end_time
            ):
                if self.trade_index % 2 == 1:
                    self.trade_trend[(order.stock_id, order.direction)] = _pred_trend
                continue

            _amount_trade_unit = self.trade_exchange.get_amount_of_trade_unit(order.factor)
            if _pred_trend == self.TREND_MID:
                _order_amount = None
                if _amount_trade_unit is None:
                    _order_amount = self.trade_amount[(order.stock_id, order.direction)] / (
                        self.trade_len - self.trade_index + 1
                    )
                elif self.trade_amount[(order.stock_id, order.direction)] >= _amount_trade_unit:
                    trade_unit_cnt = int(self.trade_amount[(order.stock_id, order.direction)] // _amount_trade_unit)
                    _order_amount = (
                        (trade_unit_cnt + self.trade_len - self.trade_index)
                        // (self.trade_len - self.trade_index + 1)
                        * _amount_trade_unit
                    )
                if order.direction == order.SELL:
                    if self.trade_amount[(order.stock_id, order.direction)] > 1e-5 and (
                        _order_amount is None or self.trade_index == self.trade_len
                    ):
                        _order_amount = self.trade_amount[(order.stock_id, order.direction)]

                if _order_amount:
                    _order = Order(
                        stock_id=order.stock_id,
                        amount=_order_amount,
                        start_time=trade_start_time,
                        end_time=trade_end_time,
                        direction=order.direction,  # 1 for buy
                        factor=order.factor,
                    )
                    order_list.append(_order)
                    # print("DEBUG AMOUNT", _order_amount, self.trade_amount[(order.stock_id, order.direction)], _amount_trade_unit)
            else:
                _order_amount = None
                if _amount_trade_unit is None:
                    _order_amount = (
                        2
                        * self.trade_amount[(order.stock_id, order.direction)]
                        / (self.trade_len - self.trade_index + 2)
                    )
                elif self.trade_amount[(order.stock_id, order.direction)] >= _amount_trade_unit:
                    trade_unit_cnt = int(self.trade_amount[(order.stock_id, order.direction)] // _amount_trade_unit)
                    _order_amount = (
                        (trade_unit_cnt + self.trade_len - self.trade_index + 1)
                        // (self.trade_len - self.trade_index + 2)
                        * 2
                        * _amount_trade_unit
                    )
                if order.direction == order.SELL:
                    if self.trade_amount[(order.stock_id, order.direction)] >= 1e-5 and (
                        _order_amount is None or self.trade_index == self.trade_len
                    ):
                        _order_amount = self.trade_amount[(order.stock_id, order.direction)]

                if _order_amount:
                    _order_amount = min(_order_amount, self.trade_amount[(order.stock_id, order.direction)])
                    if self.trade_index % 2 == 1:
                        if (
                            _pred_trend == self.TREND_SHORT
                            and order.direction == order.SELL
                            or _pred_trend == self.TREND_LONG
                            and order.direction == order.BUY
                        ):
                            _order = Order(
                                stock_id=order.stock_id,
                                amount=_order_amount,
                                start_time=trade_start_time,
                                end_time=trade_end_time,
                                direction=order.direction,  # 1 for buy
                                factor=order.factor,
                            )
                            order_list.append(_order)
                    else:
                        if (
                            _pred_trend == self.TREND_SHORT
                            and order.direction == order.BUY
                            or _pred_trend == self.TREND_LONG
                            and order.direction == order.SELL
                        ):
                            _order = Order(
                                stock_id=order.stock_id,
                                amount=_order_amount,
                                start_time=trade_start_time,
                                end_time=trade_end_time,
                                direction=order.direction,  # 1 for buy
                                factor=order.factor,
                            )
                            order_list.append(_order)
                    # print("DEBUG AMOUNT", _order_amount, self.trade_amount[(order.stock_id, order.direction)], _amount_trade_unit)
            if self.trade_index % 2 == 1:
                self.trade_trend[(order.stock_id, order.direction)] = _pred_trend

        return order_list


class SBBStrategyEMA(SBBStrategyBase):
    """
    (S)elect the (B)etter one among every two adjacent trading (B)ars to sell or buy with (EMA) signal.
    """

    def __init__(
        self,
        step_bar,
        start_time=None,
        end_time=None,
        trade_exchange=None,
        trade_order_list=[],
        instruments="csi300",
        freq="day",
        **kwargs,
    ):
        """
        Parameters
        ----------
        instruments : str, optional
            instruments of EMA signal, by default "csi300"
        freq : str, optional
            freq of EMA signal, by default "day"
            Note: `freq` may be different from `steb_bar`
        """
        super(SBBStrategyEMA, self).__init__(step_bar, start_time, end_time, trade_exchange, trade_order_list, **kwargs)
        if instruments is None:
            warnings.warn("`instruments` is not set, will load all stocks")
            self.instruments = "all"
        if isinstance(instruments, str):
            self.instruments = D.instruments(instruments)
        self.freq = freq

    def reset(self, start_time: Union[str, pd.Timestamp] = None, end_time: Union[str, pd.Timestamp] = None, **kwargs):
        """
        Reset EMA signal for trading

        Parameters
        ----------
        start_time : Union[str, pd.Timestamp], optional
            start time for trading, also used to calculate the start time of EMA signal, by default None

        end_time : Union[str, pd.Timestamp], optional
            end time for trading, also used to calculate the end time of EMA signal, by default None
        """
        super(SBBStrategyEMA, self).reset(start_time=start_time, end_time=end_time, **kwargs)
        if self.start_time and self.end_time and (start_time or end_time):
            fields = ["EMA($close, 10)-EMA($close, 20)"]
            signal_start_time, _ = self._get_calendar_time(trade_index=1, shift=1)
            _, signal_end_time = self._get_calendar_time(trade_index=self.trade_len, shift=1)
            signal_df = D.features(
                self.instruments, fields, start_time=signal_start_time, end_time=signal_end_time, freq=self.freq
            )
            signal_df = convert_index_format(signal_df)
            signal_df.columns = ["signal"]
            self.signal = {}
            for stock_id, stock_val in signal_df.groupby(level="instrument"):
                self.signal[stock_id] = stock_val

    def _pred_price_trend(self, stock_id, pred_start_time=None, pred_end_time=None):

        if stock_id not in self.signal:
            return self.TREND_MID
        else:
            _sample_signal = sample_feature(
                self.signal[stock_id], pred_start_time, pred_end_time, fields="signal", method="last"
            )
            if _sample_signal is None or _sample_signal.iloc[0] == 0:
                return self.TREND_MID
            elif _sample_signal.iloc[0] > 0:
                return self.TREND_LONG
            else:
                return self.TREND_SHORT
