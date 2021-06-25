import warnings
import numpy as np
import pandas as pd
from typing import List, Tuple, Union

from ...utils.resam import resam_ts_data
from ...data.data import D
from ...data.dataset.utils import convert_index_format
from ...strategy.base import BaseStrategy
from ...backtest.order import Order
from ...backtest.exchange import Exchange
from ...backtest.utils import CommonInfrastructure, LevelInfrastructure, TradeDecison


class TWAPStrategy(BaseStrategy):
    """TWAP Strategy for trading"""

    def __init__(
        self,
        outer_trade_decision: TradeDecison = None,
        trade_exchange: Exchange = None,
        level_infra: LevelInfrastructure = None,
        common_infra: CommonInfrastructure = None,
    ):
        """
        Parameters
        ----------
        outer_trade_decision : TradeDecison
            the trade decison of outer strategy which this startegy relies
        trade_exchange : Exchange
            exchange that provides market info, used to deal order and generate report
            - If `trade_exchange` is None, self.trade_exchange will be set with common_infra
            - It allowes different trade_exchanges is used in different executions.
            - For example:
                - In daily execution, both daily exchange and minutely are usable, but the daily exchange is recommended because it run faster.
                - In minutely execution, the daily exchange is not usable, only the minutely exchange is recommended.

        """
        super(TWAPStrategy, self).__init__(
            outer_trade_decision=outer_trade_decision, level_infra=level_infra, common_infra=common_infra
        )

        if trade_exchange is not None:
            self.trade_exchange = trade_exchange

    def reset_common_infra(self, common_infra):
        """
        Parameters
        ----------
        common_infra : CommonInfrastructure, optional
            common infrastructure for backtesting, by default None
            - It should include `trade_account`, used to get position
            - It should include `trade_exchange`, used to provide market info
        """
        super(TWAPStrategy, self).reset_common_infra(common_infra)

        if common_infra.has("trade_exchange"):
            self.trade_exchange = common_infra.get("trade_exchange")

    def reset(self, outer_trade_decision: TradeDecison = None, **kwargs):
        """
        Parameters
        ----------
        outer_trade_decision : TradeDecison, optional
        """

        super(TWAPStrategy, self).reset(outer_trade_decision=outer_trade_decision, **kwargs)
        if outer_trade_decision is not None:
            self.trade_amount = {}
            outer_order_generator = outer_trade_decision.generator()
            for order in outer_order_generator:
                self.trade_amount[order.stock_id] = order.amount

    def generate_trade_decision(self, execute_result=None):
        # get the number of trading step finished, trade_step can be [0, 1, 2, ..., trade_len - 1]
        trade_step = self.trade_calendar.get_trade_step()
        # get the total count of trading step
        trade_len = self.trade_calendar.get_trade_len()

        # update the order amount
        if execute_result is not None:
            for order, _, _, _ in execute_result:
                self.trade_amount[order.stock_id] -= order.deal_amount

        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        order_list = []
        outer_order_generator = self.outer_trade_decision.generator(only_enable=True)
        for order in outer_order_generator:
            # if not tradable, continue
            if not self.trade_exchange.is_stock_tradable(
                stock_id=order.stock_id, start_time=trade_start_time, end_time=trade_end_time
            ):
                continue
            _amount_trade_unit = self.trade_exchange.get_amount_of_trade_unit(order.factor)
            _order_amount = None
            # considering trade unit
            if _amount_trade_unit is None:
                # divide the order into equal parts, and trade one part
                _order_amount = self.trade_amount[order.stock_id] / (trade_len - trade_step)
            # without considering trade unit
            else:
                # divide the order into equal parts, and trade one part
                # calculate the total count of trade units to trade
                trade_unit_cnt = int(self.trade_amount[order.stock_id] // _amount_trade_unit)
                # calculate the amount of one part, ceil the amount
                # floor((trade_unit_cnt + trade_len - trade_step) / (trade_len - trade_step + 1)) == ceil(trade_unit_cnt / (trade_len - trade_step + 1))
                _order_amount = (
                    (trade_unit_cnt + trade_len - trade_step - 1) // (trade_len - trade_step) * _amount_trade_unit
                )

            if order.direction == order.SELL:
                # sell all amount at last
                if self.trade_amount[order.stock_id] > 1e-5 and (_order_amount < 1e-5 or trade_step == trade_len - 1):
                    _order_amount = self.trade_amount[order.stock_id]

            _order_amount = min(_order_amount, self.trade_amount[order.stock_id])

            if _order_amount > 1e-5:

                _order = Order(
                    stock_id=order.stock_id,
                    amount=_order_amount,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                    direction=order.direction,  # 1 for buy
                    factor=order.factor,
                )
                order_list.append(_order)
        return TradeDecison(order_list=order_list, ori_strategy=self)


class SBBStrategyBase(BaseStrategy):
    """
    (S)elect the (B)etter one among every two adjacent trading (B)ars to sell or buy.
    """

    TREND_MID = 0
    TREND_SHORT = 1
    TREND_LONG = 2

    def __init__(
        self,
        outer_trade_decision: TradeDecison = None,
        trade_exchange: Exchange = None,
        level_infra: LevelInfrastructure = None,
        common_infra: CommonInfrastructure = None,
    ):
        """
        Parameters
        ----------
        outer_trade_decision : TradeDecison
            the trade decison of outer strategy which this startegy relies
        trade_exchange : Exchange
            exchange that provides market info, used to deal order and generate report
            - If `trade_exchange` is None, self.trade_exchange will be set with common_infra
            - It allowes different trade_exchanges is used in different executions.
            - For example:
                - In daily execution, both daily exchange and minutely are usable, but the daily exchange is recommended because it run faster.
                - In minutely execution, the daily exchange is not usable, only the minutely exchange is recommended.
        """
        super(SBBStrategyBase, self).__init__(
            outer_trade_decision=outer_trade_decision, level_infra=level_infra, common_infra=common_infra
        )

        if trade_exchange is not None:
            self.trade_exchange = trade_exchange

    def reset_common_infra(self, common_infra):
        """
        Parameters
        ----------
        common_infra : dict, optional
            common infrastructure for backtesting, by default None
            - It should include `trade_account`, used to get position
            - It should include `trade_exchange`, used to provide market info
        """
        super(SBBStrategyBase, self).reset_common_infra(common_infra)
        if common_infra.has("trade_exchange"):
            self.trade_exchange = common_infra.get("trade_exchange")

    def reset(self, outer_trade_decision: TradeDecison = None, **kwargs):
        """
        Parameters
        ----------
        outer_trade_decision : TradeDecison, optional
        """
        super(SBBStrategyBase, self).reset(outer_trade_decision=outer_trade_decision, **kwargs)
        if outer_trade_decision is not None:
            self.trade_trend = {}
            self.trade_amount = {}
            # init the trade amount of order and  predicted trade trend
            outer_order_generator = outer_trade_decision.generator()
            for order in outer_order_generator:
                self.trade_trend[order.stock_id] = self.TREND_MID
                self.trade_amount[order.stock_id] = order.amount

    def _pred_price_trend(self, stock_id, pred_start_time=None, pred_end_time=None):
        raise NotImplementedError("pred_price_trend method is not implemented!")

    def generate_trade_decision(self, execute_result=None):
        # get the number of trading step finished, trade_step can be [0, 1, 2, ..., trade_len - 1]
        trade_step = self.trade_calendar.get_trade_step()
        # get the total count of trading step
        trade_len = self.trade_calendar.get_trade_len()

        # update the order amount
        if execute_result is not None:
            for order, _, _, _ in execute_result:
                self.trade_amount[order.stock_id] -= order.deal_amount

        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        pred_start_time, pred_end_time = self.trade_calendar.get_step_time(trade_step, shift=1)
        order_list = []
        # for each order in in self.outer_trade_decision
        outer_order_generator = self.outer_trade_decision.generator(only_enable=True)
        for order in outer_order_generator:
            # get the price trend
            if trade_step % 2 == 0:
                # in the first of two adjacent bars, predict the price trend
                _pred_trend = self._pred_price_trend(order.stock_id, pred_start_time, pred_end_time)
            else:
                # in the second of two adjacent bars, use the trend predicted in the first one
                _pred_trend = self.trade_trend[order.stock_id]
            # if not tradable, continue
            if not self.trade_exchange.is_stock_tradable(
                stock_id=order.stock_id, start_time=trade_start_time, end_time=trade_end_time
            ):
                if trade_step % 2 == 0:
                    self.trade_trend[order.stock_id] = _pred_trend
                continue
            # get amount of one trade unit
            _amount_trade_unit = self.trade_exchange.get_amount_of_trade_unit(order.factor)
            if _pred_trend == self.TREND_MID:
                _order_amount = None
                # considering trade unit
                if _amount_trade_unit is None:
                    # divide the order into equal parts, and trade one part
                    _order_amount = self.trade_amount[order.stock_id] / (trade_len - trade_step)
                # without considering trade unit
                else:
                    # divide the order into equal parts, and trade one part
                    # calculate the total count of trade units to trade
                    trade_unit_cnt = int(self.trade_amount[order.stock_id] // _amount_trade_unit)
                    # calculate the amount of one part, ceil the amount
                    # floor((trade_unit_cnt + trade_len - trade_step - 1) / (trade_len - trade_step)) == ceil(trade_unit_cnt / (trade_len - trade_step))
                    _order_amount = (
                        (trade_unit_cnt + trade_len - trade_step - 1) // (trade_len - trade_step) * _amount_trade_unit
                    )
                if order.direction == order.SELL:
                    # sell all amount at last
                    if self.trade_amount[order.stock_id] > 1e-5 and (
                        _order_amount < 1e-5 or trade_step == trade_len - 1
                    ):
                        _order_amount = self.trade_amount[order.stock_id]

                _order_amount = min(_order_amount, self.trade_amount[order.stock_id])

                if _order_amount > 1e-5:
                    _order = Order(
                        stock_id=order.stock_id,
                        amount=_order_amount,
                        start_time=trade_start_time,
                        end_time=trade_end_time,
                        direction=order.direction,
                        factor=order.factor,
                    )
                    order_list.append(_order)

            else:
                _order_amount = None
                # considering trade unit
                if _amount_trade_unit is None:
                    # N trade day left, divide the order into N + 1 parts, and trade 2 parts
                    _order_amount = 2 * self.trade_amount[order.stock_id] / (trade_len - trade_step + 1)
                # without considering trade unit
                else:
                    # cal how many trade unit
                    trade_unit_cnt = int(self.trade_amount[order.stock_id] // _amount_trade_unit)
                    # N trade day left, divide the order into N + 1 parts, and trade 2 parts
                    _order_amount = (
                        (trade_unit_cnt + trade_len - trade_step)
                        // (trade_len - trade_step + 1)
                        * 2
                        * _amount_trade_unit
                    )
                if order.direction == order.SELL:
                    # sell all amount at last
                    if self.trade_amount[order.stock_id] > 1e-5 and (
                        _order_amount < 1e-5 or trade_step == trade_len - 1
                    ):
                        _order_amount = self.trade_amount[order.stock_id]

                _order_amount = min(_order_amount, self.trade_amount[order.stock_id])

                if _order_amount > 1e-5:
                    if trade_step % 2 == 0:
                        # in the first one of two adjacent bars
                        # if look short on the price, sell the stock more
                        # if look long on the price, buy the stock more
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
                        # in the second one of two adjacent bars
                        # if look short on the price, buy the stock more
                        # if look long on the price, sell the stock more
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

            if trade_step % 2 == 0:
                # in the first one of two adjacent bars, store the trend for the second one to use
                self.trade_trend[order.stock_id] = _pred_trend

        return TradeDecison(order_list=order_list, ori_strategy=self)


class SBBStrategyEMA(SBBStrategyBase):
    """
    (S)elect the (B)etter one among every two adjacent trading (B)ars to sell or buy with (EMA) signal.
    """

    def __init__(
        self,
        outer_trade_decision: TradeDecison = None,
        instruments: Union[List, str] = "csi300",
        freq: str = "day",
        trade_exchange: Exchange = None,
        level_infra: LevelInfrastructure = None,
        common_infra: CommonInfrastructure = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        instruments : Union[List, str], optional
            instruments of EMA signal, by default "csi300"
        freq : str, optional
            freq of EMA signal, by default "day"
            Note: `freq` may be different from `time_per_step`
        """
        if instruments is None:
            warnings.warn("`instruments` is not set, will load all stocks")
            self.instruments = "all"
        if isinstance(instruments, str):
            self.instruments = D.instruments(instruments)
        self.freq = freq
        super(SBBStrategyEMA, self).__init__(outer_trade_decision, trade_exchange, level_infra, common_infra, **kwargs)

    def _reset_signal(self):
        trade_len = self.trade_calendar.get_trade_len()
        fields = ["EMA($close, 10)-EMA($close, 20)"]
        signal_start_time, _ = self.trade_calendar.get_step_time(trade_step=0, shift=1)
        _, signal_end_time = self.trade_calendar.get_step_time(trade_step=trade_len - 1, shift=1)
        signal_df = D.features(
            self.instruments, fields, start_time=signal_start_time, end_time=signal_end_time, freq=self.freq
        )
        signal_df = convert_index_format(signal_df)
        signal_df.columns = ["signal"]
        self.signal = {}

        if not signal_df.empty:
            for stock_id, stock_val in signal_df.groupby(level="instrument"):
                self.signal[stock_id] = stock_val

    def reset_level_infra(self, level_infra):
        """
        reset level-shared infra
        - After reset the trade calendar, the signal will be changed
        """
        if not hasattr(self, "level_infra"):
            self.level_infra = level_infra
        else:
            self.level_infra.update(level_infra)

        if level_infra.has("trade_calendar"):
            self.trade_calendar = level_infra.get("trade_calendar")
            self._reset_signal()

    def _pred_price_trend(self, stock_id, pred_start_time=None, pred_end_time=None):
        # if no signal, return mid trend
        if stock_id not in self.signal:
            return self.TREND_MID
        else:
            _sample_signal = resam_ts_data(
                self.signal[stock_id]["signal"], pred_start_time, pred_end_time, method="last"
            )
            # if EMA signal == 0 or None, return mid trend
            if _sample_signal is None or _sample_signal.iloc[0] == 0:
                return self.TREND_MID
            # if EMA signal > 0, return long trend
            elif _sample_signal.iloc[0] > 0:
                return self.TREND_LONG
            # if EMA signal < 0, return short trend
            else:
                return self.TREND_SHORT


class ACStrategy(BaseStrategy):
    def __init__(
        self,
        lamb: float = 1e-6,
        eta: float = 2.5e-6,
        window_size: int = 20,
        outer_trade_decision: TradeDecison = None,
        instruments: Union[List, str] = "csi300",
        freq: str = "day",
        trade_exchange: Exchange = None,
        level_infra: LevelInfrastructure = None,
        common_infra: CommonInfrastructure = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        instruments : Union[List, str], optional
            instruments of Volatility, by default "csi300"
        freq : str, optional
            freq of Volatility, by default "day"
            Note: `freq` may be different from `time_per_step`
        """
        self.lamb = lamb
        self.eta = eta
        self.window_size = window_size
        if instruments is None:
            warnings.warn("`instruments` is not set, will load all stocks")
            self.instruments = "all"
        if isinstance(instruments, str):
            self.instruments = D.instruments(instruments)
        self.freq = freq
        super(ACStrategy, self).__init__(outer_trade_decision, level_infra, common_infra, **kwargs)

        if trade_exchange is not None:
            self.trade_exchange = trade_exchange

    def _reset_signal(self):
        trade_len = self.trade_calendar.get_trade_len()
        fields = [
            f"Power(Sum(Power(Log($close/Ref($close, 1)), 2), {self.window_size})/{self.window_size - 1}-Power(Sum(Log($close/Ref($close, 1)), {self.window_size}), 2)/({self.window_size}*{self.window_size - 1}), 0.5)"
        ]
        signal_start_time, _ = self.trade_calendar.get_step_time(trade_step=0, shift=1)
        _, signal_end_time = self.trade_calendar.get_step_time(trade_step=trade_len - 1, shift=1)
        signal_df = D.features(
            self.instruments, fields, start_time=signal_start_time, end_time=signal_end_time, freq=self.freq
        )
        signal_df = convert_index_format(signal_df)
        signal_df.columns = ["volatility"]
        self.signal = {}

        if not signal_df.empty:
            for stock_id, stock_val in signal_df.groupby(level="instrument"):
                self.signal[stock_id] = stock_val

    def reset_common_infra(self, common_infra):
        """
        Parameters
        ----------
        common_infra : CommonInfrastructure, optional
            common infrastructure for backtesting, by default None
            - It should include `trade_account`, used to get position
            - It should include `trade_exchange`, used to provide market info
        """
        super(ACStrategy, self).reset_common_infra(common_infra)

        if common_infra.has("trade_exchange"):
            self.trade_exchange = common_infra.get("trade_exchange")

    def reset_level_infra(self, level_infra):
        """
        reset level-shared infra
        - After reset the trade calendar, the signal will be changed
        """
        if not hasattr(self, "level_infra"):
            self.level_infra = level_infra
        else:
            self.level_infra.update(level_infra)

        if level_infra.has("trade_calendar"):
            self.trade_calendar = level_infra.get("trade_calendar")
            self._reset_signal()

    def reset(self, outer_trade_decision: TradeDecison = None, **kwargs):
        """
        Parameters
        ----------
        outer_trade_decision : TradeDecison, optional
        """
        super(ACStrategy, self).reset(outer_trade_decision=outer_trade_decision, **kwargs)
        if outer_trade_decision is not None:
            self.trade_amount = {}
            # init the trade amount of order and  predicted trade trend
            outer_order_generator = outer_trade_decision.generator()
            for order in outer_order_generator:
                self.trade_amount[order.stock_id] = order.amount

    def generate_trade_decision(self, execute_result=None):
        # get the number of trading step finished, trade_step can be [0, 1, 2, ..., trade_len - 1]
        trade_step = self.trade_calendar.get_trade_step()
        # get the total count of trading step
        trade_len = self.trade_calendar.get_trade_len()
        # update outer trade decision
        self.outer_trade_decision.update(self.trade_calendar)

        # update the order amount
        if execute_result is not None:
            for order, _, _, _ in execute_result:
                self.trade_amount[order.stock_id] -= order.deal_amount

        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        pred_start_time, pred_end_time = self.trade_calendar.get_step_time(trade_step, shift=1)
        order_list = []
        outer_order_generator = self.outer_trade_decision.generator(only_enable=True)
        for order in outer_order_generator:
            # if not tradable, continue
            if not self.trade_exchange.is_stock_tradable(
                stock_id=order.stock_id, start_time=trade_start_time, end_time=trade_end_time
            ):
                continue
            _order_amount = None
            # considering trade unit

            sig_sam = (
                resam_ts_data(self.signal[order.stock_id]["volatility"], pred_start_time, pred_end_time, method="last")
                if order.stock_id in self.signal
                else None
            )

            if sig_sam is None or sig_sam.iloc[0] is None:
                # no signal, TWAP
                _amount_trade_unit = self.trade_exchange.get_amount_of_trade_unit(order.factor)
                if _amount_trade_unit is None:
                    # divide the order into equal parts, and trade one part
                    _order_amount = self.trade_amount[order.stock_id] / (trade_len - trade_step)
                else:
                    # divide the order into equal parts, and trade one part
                    # calculate the total count of trade units to trade
                    trade_unit_cnt = int(self.trade_amount[order.stock_id] // _amount_trade_unit)
                    # calculate the amount of one part, ceil the amount
                    # floor((trade_unit_cnt + trade_len - trade_step - 1) / (trade_len - trade_step)) == ceil(trade_unit_cnt / (trade_len - trade_step))
                    _order_amount = (
                        (trade_unit_cnt + trade_len - trade_step - 1) // (trade_len - trade_step) * _amount_trade_unit
                    )
            else:
                # VA strategy
                kappa_tild = self.lamb / self.eta * sig_sam.iloc[0] * sig_sam.iloc[0]
                kappa = np.arccosh(kappa_tild / 2 + 1)
                amount_ratio = (
                    np.sinh(kappa * (trade_len - trade_step)) - np.sinh(kappa * (trade_len - trade_step - 1))
                ) / np.sinh(kappa * trade_len)
                _order_amount = order.amount * amount_ratio
                _order_amount = self.trade_exchange.round_amount_by_trade_unit(_order_amount, order.factor)

            if order.direction == order.SELL:
                # sell all amount at last
                if self.trade_amount[order.stock_id] > 1e-5 and (_order_amount < 1e-5 or trade_step == trade_len - 1):
                    _order_amount = self.trade_amount[order.stock_id]

            _order_amount = min(_order_amount, self.trade_amount[order.stock_id])

            if _order_amount > 1e-5:

                _order = Order(
                    stock_id=order.stock_id,
                    amount=_order_amount,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                    direction=order.direction,  # 1 for buy
                    factor=order.factor,
                )
                order_list.append(_order)
        return TradeDecison(order_list=order_list, ori_strategy=self)


class RandomOrderStrategy(BaseStrategy):

    def __init__(self,
                 time_range: Tuple = ("9:30", "15:00"),  # The range is closed on both left and right.
                 sample_ratio: float = 1.,
                 volume_ratio: float = 0.01,
                 market: str = "all",
                 *args,
                 **kwargs):
        """
        Parameters
        ----------
        time_range : Tuple
            the intra day time range of the orders
            the left and right is closed.
            # TODO: this is a time_range level limitation. We'll implement a more detailed limitation later.
        sample_ratio : float
            the ratio of all orders are sampled
        volume_ratio : float
            the volume of the total day
            raito of the total volume of a specific day
        market : str
            stock pool for sampling
        """

        super().__init__(*args, **kwargs)
        self.time_range = time_range
        self.sample_ratio = sample_ratio
        self.volume_ratio = volume_ratio
        self.market = market
        exch: Exchange = self.common_infra.get("exchange")
        self.volume = D.features(D.instruments("market"), ["Mean($volume, 10)"], start_time=exch.start_time, end_time=exch.end_time)

    def generate_trade_decision(self, execute_result=None):
        return super().generate_trade_decision(execute_result=execute_result)
