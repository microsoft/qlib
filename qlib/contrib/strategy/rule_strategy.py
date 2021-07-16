from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from typing import IO, List, Tuple, Union
from qlib.data.dataset.utils import convert_index_format

from qlib.utils import lazy_sort_index

from ...utils.resam import resam_ts_data, ts_data_last
from ...data.data import D
from ...strategy.base import BaseStrategy
from ...backtest.order import BaseTradeDecision, Order, TradeDecisionWO, TradeRange
from ...backtest.exchange import Exchange, OrderHelper
from ...backtest.utils import CommonInfrastructure, LevelInfrastructure
from qlib.utils.file import get_io_object
from qlib.backtest.utils import get_start_end_idx


class TWAPStrategy(BaseStrategy):
    """TWAP Strategy for trading"""

    def __init__(
        self,
        outer_trade_decision: BaseTradeDecision = None,
        trade_exchange: Exchange = None,
        level_infra: LevelInfrastructure = None,
        common_infra: CommonInfrastructure = None,
    ):
        """
        Parameters
        ----------
        outer_trade_decision : BaseTradeDecision
            the trade decision of outer strategy which this startegy relies
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

    def reset(self, outer_trade_decision: BaseTradeDecision = None, **kwargs):
        """
        Parameters
        ----------
        outer_trade_decision : BaseTradeDecision, optional
        """

        super(TWAPStrategy, self).reset(outer_trade_decision=outer_trade_decision, **kwargs)
        if outer_trade_decision is not None:
            self.trade_amount = {}
            for order in outer_trade_decision.get_decision():
                self.trade_amount[order.stock_id] = order.amount

    def generate_trade_decision(self, execute_result=None):
        # strategy is not available. Give an empty decision
        if len(self.outer_trade_decision.get_decision()) == 0:
            return TradeDecisionWO(order_list=[], strategy=self)

        # get the number of trading step finished, trade_step can be [0, 1, 2, ..., trade_len - 1]
        trade_step = self.trade_calendar.get_trade_step()
        # get the total count of trading step
        start_idx, end_idx = get_start_end_idx(self.trade_calendar, self.outer_trade_decision)
        trade_len = end_idx - start_idx + 1

        if trade_step < start_idx or trade_step > end_idx:
            # It is not time to start trading or trading has ended.
            return TradeDecisionWO(order_list=[], strategy=self)

        rel_trade_step = trade_step - start_idx  # trade_step relative to start_idx

        # update the order amount
        if execute_result is not None:
            for order, _, _, _ in execute_result:
                self.trade_amount[order.stock_id] -= order.deal_amount

        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        order_list = []
        for order in self.outer_trade_decision.get_decision():
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
                _order_amount = self.trade_amount[order.stock_id] / (trade_len - rel_trade_step)
            # without considering trade unit
            else:
                # divide the order into equal parts, and trade one part
                # calculate the total count of trade units to trade
                trade_unit_cnt = int(self.trade_amount[order.stock_id] // _amount_trade_unit)
                # calculate the amount of one part, ceil the amount
                # floor((trade_unit_cnt + trade_len - rel_trade_step) / (trade_len - rel_trade_step + 1)) == ceil(trade_unit_cnt / (trade_len - rel_trade_step + 1))
                _order_amount = (
                    (trade_unit_cnt + trade_len - rel_trade_step - 1)
                    // (trade_len - rel_trade_step)
                    * _amount_trade_unit
                )

            if order.direction == order.SELL:
                # sell all amount at last
                if self.trade_amount[order.stock_id] > 1e-5 and (
                    _order_amount < 1e-5 or rel_trade_step == trade_len - 1
                ):
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
        return TradeDecisionWO(order_list=order_list, strategy=self)


class SBBStrategyBase(BaseStrategy):
    """
    (S)elect the (B)etter one among every two adjacent trading (B)ars to sell or buy.
    """

    TREND_MID = 0
    TREND_SHORT = 1
    TREND_LONG = 2

    # TODO:
    # 1. Supporting leverage the get_range_limit result from the decision
    # 2. Supporting alter_outer_trade_decision
    # 3. Supporting checking the availability of trade decision

    def __init__(
        self,
        outer_trade_decision: BaseTradeDecision = None,
        trade_exchange: Exchange = None,
        level_infra: LevelInfrastructure = None,
        common_infra: CommonInfrastructure = None,
    ):
        """
        Parameters
        ----------
        outer_trade_decision : BaseTradeDecision
            the trade decision of outer strategy which this startegy relies
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

    def reset(self, outer_trade_decision: BaseTradeDecision = None, **kwargs):
        """
        Parameters
        ----------
        outer_trade_decision : BaseTradeDecision, optional
        """
        super(SBBStrategyBase, self).reset(outer_trade_decision=outer_trade_decision, **kwargs)
        if outer_trade_decision is not None:
            self.trade_trend = {}
            self.trade_amount = {}
            # init the trade amount of order and  predicted trade trend
            for order in outer_trade_decision.get_decision():
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
        for order in self.outer_trade_decision.get_decision():
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

        return TradeDecisionWO(order_list, self)


class SBBStrategyEMA(SBBStrategyBase):
    """
    (S)elect the (B)etter one among every two adjacent trading (B)ars to sell or buy with (EMA) signal.
    """

    # TODO:
    # 1. Supporting leverage the get_range_limit result from the decision
    # 2. Supporting alter_outer_trade_decision
    # 3. Supporting checking the availability of trade decision

    def __init__(
        self,
        outer_trade_decision: BaseTradeDecision = None,
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
        signal_df.columns = ["signal"]
        self.signal = {}

        if not signal_df.empty:
            for stock_id, stock_val in signal_df.groupby(level="instrument"):
                self.signal[stock_id] = stock_val["signal"].droplevel(level="instrument")

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
                self.signal[stock_id],
                pred_start_time,
                pred_end_time,
                method=ts_data_last,
            )
            # if EMA signal == 0 or None, return mid trend
            if _sample_signal is None or np.isnan(_sample_signal) or _sample_signal == 0:
                return self.TREND_MID
            # if EMA signal > 0, return long trend
            elif _sample_signal > 0:
                return self.TREND_LONG
            # if EMA signal < 0, return short trend
            else:
                return self.TREND_SHORT


class ACStrategy(BaseStrategy):
    # TODO:
    # 1. Supporting leverage the get_range_limit result from the decision
    # 2. Supporting alter_outer_trade_decision
    # 3. Supporting checking the availability of trade decision
    def __init__(
        self,
        lamb: float = 1e-6,
        eta: float = 2.5e-6,
        window_size: int = 20,
        outer_trade_decision: BaseTradeDecision = None,
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
        signal_df.columns = ["volatility"]
        self.signal = {}

        if not signal_df.empty:
            for stock_id, stock_val in signal_df.groupby(level="instrument"):
                self.signal[stock_id] = stock_val["volatility"].droplevel(level="instrument")

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

    def reset(self, outer_trade_decision: BaseTradeDecision = None, **kwargs):
        """
        Parameters
        ----------
        outer_trade_decision : BaseTradeDecision, optional
        """
        super(ACStrategy, self).reset(outer_trade_decision=outer_trade_decision, **kwargs)
        if outer_trade_decision is not None:
            self.trade_amount = {}
            # init the trade amount of order and  predicted trade trend
            for order in outer_trade_decision.get_decision():
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
        pred_start_time, pred_end_time = self.trade_calendar.get_step_time(trade_step, shift=1)
        order_list = []
        for order in self.outer_trade_decision.get_decision():
            # if not tradable, continue
            if not self.trade_exchange.is_stock_tradable(
                stock_id=order.stock_id, start_time=trade_start_time, end_time=trade_end_time
            ):
                continue
            _order_amount = None
            # considering trade unit

            sig_sam = (
                resam_ts_data(self.signal[order.stock_id], pred_start_time, pred_end_time, method=ts_data_last)
                if order.stock_id in self.signal
                else None
            )

            if sig_sam is None or np.isnan(sig_sam):
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
                kappa_tild = self.lamb / self.eta * sig_sam * sig_sam
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
        return TradeDecisionWO(order_list, self)


class RandomOrderStrategy(BaseStrategy):
    def __init__(
        self,
        trade_range: Union[Tuple[int, int], TradeRange],  # The range is closed on both left and right.
        sample_ratio: float = 1.0,
        volume_ratio: float = 0.01,
        market: str = "all",
        direction: int = Order.BUY,
        *args,
        **kwargs,
    ):
        """
        Parameters
        ----------
        trade_range : Tuple
            please refer to the `trade_range` parameter of BaseStrategy
        sample_ratio : float
            the ratio of all orders are sampled
        volume_ratio : float
            the volume of the total day
            raito of the total volume of a specific day
        market : str
            stock pool for sampling
        """

        super().__init__(*args, **kwargs)
        self.sample_ratio = sample_ratio
        self.volume_ratio = volume_ratio
        self.market = market
        self.direction = direction
        exch: Exchange = self.common_infra.get("trade_exchange")
        # TODO: this can't be online
        self.volume = D.features(
            D.instruments(market), ["Mean(Ref($volume, 1), 10)"], start_time=exch.start_time, end_time=exch.end_time
        )
        self.volume_df = self.volume.iloc[:, 0].unstack()
        self.trade_range = trade_range

    def generate_trade_decision(self, execute_result=None):
        trade_step = self.trade_calendar.get_trade_step()
        step_time_start, step_time_end = self.trade_calendar.get_step_time(trade_step)

        order_list = []
        if step_time_start in self.volume_df:
            for stock_id, volume in self.volume_df[step_time_start].dropna().sample(frac=self.sample_ratio).items():
                order_list.append(
                    self.common_infra.get("trade_exchange")
                    .get_order_helper()
                    .create(
                        code=stock_id,
                        amount=volume * self.volume_ratio,
                        start_time=step_time_start,
                        end_time=step_time_end,
                        direction=self.direction,
                    )
                )
        return TradeDecisionWO(order_list, self, self.trade_range)


class FileOrderStrategy(BaseStrategy):
    """
    Motivation:
    - This class provides an interface for user to read orders from csv files.
    """

    def __init__(
        self, file: Union[IO, str, Path], trade_range: Union[Tuple[int, int], TradeRange] = None, *args, **kwargs
    ):
        """

        Parameters
        ----------
        file : Union[IO, str, Path]
            this parameters will specify the info of expected orders

            Here is an example of the content

            1) Amount (**adjusted**) based strategy

                datetime,instrument,amount,direction
                20200102,  SH600519,  1000,     sell
                20200103,  SH600519,  1000,      buy
                20200106,  SH600519,  1000,     sell

        trade_range : Tuple[int, int]
            the intra day time index range of the orders
            the left and right is closed.

            If you want to get the trade_range in intra-day
            - `qlib/utils/time.py:def get_day_min_idx_range` can help you create the index range easier
            # TODO: this is a trade_range level limitation. We'll implement a more detailed limitation later.

        """
        super().__init__(*args, **kwargs)
        with get_io_object(file) as f:
            self.order_df = pd.read_csv(f, dtype={"datetime": np.str})

        self.order_df["datetime"] = self.order_df["datetime"].apply(pd.Timestamp)
        self.order_df = self.order_df.set_index(["datetime", "instrument"])

        # make sure the datetime is the first level for fast indexing
        self.order_df = lazy_sort_index(convert_index_format(self.order_df, level="datetime"))
        self.trade_range = trade_range

    def generate_trade_decision(self, execute_result=None) -> TradeDecisionWO:
        """
        Parameters
        ----------
        execute_result :
            execute_result will be ignored in FileOrderStrategy
        """
        oh: OrderHelper = self.common_infra.get("trade_exchange").get_order_helper()
        tc = self.trade_calendar
        step = tc.get_trade_step()
        start, end = tc.get_step_time(step)
        # CONVERSION: the bar is indexed by the time
        try:
            df = self.order_df.loc(axis=0)[start]
        except KeyError:
            return TradeDecisionWO([], self)
        else:
            order_list = []
            for idx, row in df.iterrows():
                order_list.append(
                    oh.create(
                        code=idx,
                        amount=row["amount"],
                        direction=Order.parse_dir(row["direction"]),
                        start_time=start,
                        end_time=end,
                    )
                )
            return TradeDecisionWO(order_list, self, self.trade_range)
