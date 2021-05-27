import warnings

from ...utils.resam import resam_ts_data
from ...data.data import D
from ...data.dataset.utils import convert_index_format
from ...strategy.base import BaseStrategy
from ...backtest.order import Order
from ...backtest.exchange import Exchange


class TWAPStrategy(BaseStrategy):
    """TWAP Strategy for trading"""

    def __init__(
        self,
        outer_trade_decision: object = None,
        trade_exchange: Exchange = None,
        level_infra: dict = {},
        common_infra: dict = {},
    ):
        """
        Parameters
        ----------
        trade_exchange : Exchange
            exchange that provides market info, used to deal order and generate report
            - If `trade_exchange` is None, self.trade_exchange will be set with common_infra
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
        common_infra : dict, optional
            common infrastructure for backtesting, by default None
            - It should include `trade_account`, used to get position
            - It should include `trade_exchange`, used to provide market info
        """
        super(TWAPStrategy, self).reset_common_infra(common_infra)
        if common_infra is not None:
            if "trade_exchange" in common_infra:
                self.trade_exchange = common_infra.get("trade_exchange")

    def reset(self, outer_trade_decision: object = None, **kwargs):
        """
        Parameters
        ----------
        outer_trade_decision : object, optional
        """

        super(TWAPStrategy, self).reset(outer_trade_decision=outer_trade_decision, common_infra=common_infra, **kwargs)
        if outer_trade_decision is not None:
            self.trade_amount = {}
            for order in outer_trade_decision:
                self.trade_amount[(order.stock_id, order.direction)] = order.amount

    def generate_trade_decision(self, execute_result=None):

        # update the order amount
        if execute_result is not None:
            for order, _, _, _ in execute_result:
                self.trade_amount[(order.stock_id, order.direction)] -= order.deal_amount

        # get the number of trading step finished, trade_step can be [0, 1, 2, ..., trade_len - 1]
        trade_step = self.trade_calendar.get_trade_step()
        # get the total count of trading step
        trade_len = self.trade_calendar.get_trade_len()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        order_list = []
        for order in self.outer_trade_decision:
            if not self.trade_exchange.is_stock_tradable(
                stock_id=order.stock_id, start_time=trade_start_time, end_time=trade_end_time
            ):
                continue
            _amount_trade_unit = self.trade_exchange.get_amount_of_trade_unit(order.factor)
            _order_amount = None
            # consider trade unit
            if _amount_trade_unit is None:
                # divide the order equally
                _order_amount = self.trade_amount[(order.stock_id, order.direction)] / (trade_len - trade_step + 1)
            # without considering trade unit
            elif self.trade_amount[(order.stock_id, order.direction)] >= _amount_trade_unit:
                # divide the order equally
                # floor((trade_unit_cnt + trade_len - trade_step) / (trade_len - trade_step + 1)) == ceil(trade_unit_cnt / (trade_len - trade_step + 1))
                trade_unit_cnt = int(self.trade_amount[(order.stock_id, order.direction)] // _amount_trade_unit)
                _order_amount = (
                    (trade_unit_cnt + trade_len - trade_step) // (trade_len - trade_step + 1) * _amount_trade_unit
                )

            if order.direction == order.SELL:
                # sell all amount at last
                if self.trade_amount[(order.stock_id, order.direction)] > 1e-5 and (
                    _order_amount is None or trade_step == trade_len
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


class SBBStrategyBase(BaseStrategy):
    """
    (S)elect the (B)etter one among every two adjacent trading (B)ars to sell or buy.
    """

    TREND_MID = 0
    TREND_SHORT = 1
    TREND_LONG = 2

    def __init__(
        self,
        outer_trade_decision: object = None,
        trade_exchange: Exchange = None,
        level_infra: dict = {},
        common_infra: dict = {},
    ):
        """
        Parameters
        ----------
        trade_exchange : Exchange
            exchange that provides market info, used to deal order and generate report
            - If `trade_exchange` is None, self.trade_exchange will be set with common_infra
        """
        super(SBBStrategyBase, self).__init__(
            outer_trade_decision=outer_trade_decision, level_infra=level_infra, common_infra=common_infra
        )

        if trade_exchange is not None:
            self.trade_exchange = trade_exchange

    def reset_common_infra(self, common_infra):
        super(SBBStrategyBase, self).reset_common_infra(common_infra)
        if common_infra is not None:
            if "trade_exchange" in common_infra:
                self.trade_exchange = common_infra.get("trade_exchange")

    def reset(self, outer_trade_decision=None, **kwargs):
        """
        Parameters
        ----------
        outer_trade_decision : object, optional
        common_infra : None, optional
            common infrastructure for backtesting, by default None
            - It should include `trade_account`, used to get position
            - It should include `trade_exchange`, used to provide market info
        """
        super(SBBStrategyBase, self).reset(outer_trade_decision=outer_trade_decision, **kwargs)
        if outer_trade_decision is not None:
            self.trade_trend = {}
            self.trade_amount = {}
            # init the trade amount of order and  predicted trade trend
            for order in outer_trade_decision:
                self.trade_trend[(order.stock_id, order.direction)] = self.TREND_MID
                self.trade_amount[(order.stock_id, order.direction)] = order.amount

    def _pred_price_trend(self, stock_id, pred_start_time=None, pred_end_time=None):
        raise NotImplementedError("pred_price_trend method is not implemented!")

    def generate_trade_decision(self, execute_result=None):

        # update the order amount
        if execute_result is not None:
            for order, _, _, _ in execute_result:
                self.trade_amount[(order.stock_id, order.direction)] -= order.deal_amount
        # get the number of trading step finished, trade_step can be [0, 1, 2, ..., trade_len - 1]
        trade_step = self.trade_calendar.get_trade_step()
        # get the total count of trading step
        trade_len = self.trade_calendar.get_trade_len()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        pred_start_time, pred_end_time = self.trade_calendar.get_step_time(trade_step, shift=1)
        order_list = []
        # for each order in in self.outer_trade_decision
        for order in self.outer_trade_decision:
            # predict the price trend
            if trade_step % 2 == 0:
                _pred_trend = self._pred_price_trend(order.stock_id, pred_start_time, pred_end_time)
            else:
                _pred_trend = self.trade_trend[(order.stock_id, order.direction)]
            # if not tradable, continue
            if not self.trade_exchange.is_stock_tradable(
                stock_id=order.stock_id, start_time=trade_start_time, end_time=trade_end_time
            ):
                if trade_step % 2 == 0:
                    self.trade_trend[(order.stock_id, order.direction)] = _pred_trend
                continue
            # get amount of one trade unit
            _amount_trade_unit = self.trade_exchange.get_amount_of_trade_unit(order.factor)
            if _pred_trend == self.TREND_MID:
                _order_amount = None
                # considering trade unit
                if _amount_trade_unit is None:
                    # divide the order equally
                    _order_amount = self.trade_amount[(order.stock_id, order.direction)] / (trade_len - trade_step)
                # without considering trade unit
                elif self.trade_amount[(order.stock_id, order.direction)] >= _amount_trade_unit:
                    # cal how many trade unit
                    trade_unit_cnt = int(self.trade_amount[(order.stock_id, order.direction)] // _amount_trade_unit)
                    # divide the order equally
                    # floor((trade_unit_cnt + trade_len - trade_step - 1) / (trade_len - trade_step)) == ceil(trade_unit_cnt / (trade_len - trade_step))
                    _order_amount = (
                        (trade_unit_cnt + trade_len - trade_step - 1) // (trade_len - trade_step) * _amount_trade_unit
                    )
                if order.direction == order.SELL:
                    # sell all amount at last
                    if self.trade_amount[(order.stock_id, order.direction)] > 1e-5 and (
                        _order_amount is None or trade_step == trade_len - 1
                    ):
                        _order_amount = self.trade_amount[(order.stock_id, order.direction)]

                if _order_amount:
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
                    _order_amount = (
                        2 * self.trade_amount[(order.stock_id, order.direction)] / (trade_len - trade_step + 1)
                    )
                # without considering trade unit
                elif self.trade_amount[(order.stock_id, order.direction)] >= _amount_trade_unit:
                    # cal how many trade unit
                    trade_unit_cnt = int(self.trade_amount[(order.stock_id, order.direction)] // _amount_trade_unit)
                    # N trade day left, divide the order into N + 1 parts, and trade 2 parts
                    _order_amount = (
                        (trade_unit_cnt + trade_len - trade_step)
                        // (trade_len - trade_step + 1)
                        * 2
                        * _amount_trade_unit
                    )
                if order.direction == order.SELL:
                    # sell all amount at last
                    if self.trade_amount[(order.stock_id, order.direction)] >= 1e-5 and (
                        _order_amount is None or trade_step == trade_len - 1
                    ):
                        _order_amount = self.trade_amount[(order.stock_id, order.direction)]

                if _order_amount:
                    _order_amount = min(_order_amount, self.trade_amount[(order.stock_id, order.direction)])
                    if trade_step % 2 == 0:
                        # in the first of two adjacent bar
                        # if look short on the price, sell the stock more
                        # if look long on the price, sell the stock more
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
                        # in the second of two adjacent bar
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
                self.trade_trend[(order.stock_id, order.direction)] = _pred_trend

        return order_list


class SBBStrategyEMA(SBBStrategyBase):
    """
    (S)elect the (B)etter one among every two adjacent trading (B)ars to sell or buy with (EMA) signal.
    """

    def __init__(
        self,
        outer_trade_decision=[],
        instruments="csi300",
        freq="day",
        trade_exchange: Exchange = None,
        level_infra={},
        common_infra={},
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

        if "trade_calendar" in level_infra:
            self.trade_calendar = level_infra.get("trade_calendar")
            self._reset_signal()

    def _pred_price_trend(self, stock_id, pred_start_time=None, pred_end_time=None):

        if stock_id not in self.signal:
            return self.TREND_MID
        else:
            _sample_signal = resam_ts_data(
                self.signal[stock_id]["signal"], pred_start_time, pred_end_time, method="last"
            )
            if _sample_signal is None or _sample_signal.iloc[0] == 0:
                return self.TREND_MID
            elif _sample_signal.iloc[0] > 0:
                return self.TREND_LONG
            else:
                return self.TREND_SHORT
