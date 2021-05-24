import warnings

from ...utils.resam import resam_ts_data
from ...data.data import D
from ...data.dataset.utils import convert_index_format
from ...strategy.base import RuleStrategy
from ..backtest.order import Order


class TWAPStrategy(RuleStrategy):
    """TWAP Strategy for trading"""

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

    def reset(self, rely_trade_decision: object = None, **kwargs):
        """
        Parameters
        ----------
        rely_trade_decision : object, optional
        """

        super(TWAPStrategy, self).reset(rely_trade_decision=rely_trade_decision, common_infra=common_infra, **kwargs)
        if rely_trade_decision is not None:
            self.trade_amount = {}
            for order in rely_trade_decision:
                self.trade_amount[(order.stock_id, order.direction)] = order.amount

    def generate_trade_decision(self, execute_state):

        # update the order amount
        trade_info = execute_state
        for order, _, _, _ in trade_info:
            self.trade_amount[(order.stock_id, order.direction)] -= order.deal_amount

        trade_index = self.trade_calendar.get_trade_index()
        trade_len = self.trade_calendar.get_trade_len()
        trade_start_time, trade_end_time = self.trade_calendar.get_calendar_time(trade_index)
        order_list = []
        for order in self.rely_trade_decision:
            if not self.trade_exchange.is_stock_tradable(
                stock_id=order.stock_id, start_time=trade_start_time, end_time=trade_end_time
            ):
                continue
            _amount_trade_unit = self.trade_exchange.get_amount_of_trade_unit(order.factor)
            _order_amount = None
            # consider trade unit
            if _amount_trade_unit is None:
                # split the order equally
                _order_amount = self.trade_amount[(order.stock_id, order.direction)] / (trade_len - trade_index + 1)
            # without considering trade unit
            elif self.trade_amount[(order.stock_id, order.direction)] >= _amount_trade_unit:
                # split the order equally
                # floor((trade_unit_cnt + trade_len - trade_index) / (trade_len - trade_index + 1)) == ceil(trade_unit_cnt / (trade_len - trade_index + 1))
                trade_unit_cnt = int(self.trade_amount[(order.stock_id, order.direction)] // _amount_trade_unit)
                _order_amount = (
                    (trade_unit_cnt + trade_len - trade_index) // (trade_len - trade_index + 1) * _amount_trade_unit
                )

            if order.direction == order.SELL:
                # sell all amount at last
                if self.trade_amount[(order.stock_id, order.direction)] > 1e-5 and (
                    _order_amount is None or trade_index == trade_len
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


class SBBStrategyBase(RuleStrategy):
    """
    (S)elect the (B)etter one among every two adjacent trading (B)ars to sell or buy.
    """

    TREND_MID = 0
    TREND_SHORT = 1
    TREND_LONG = 2

    def reset_common_infra(self, common_infra):
        super(SBBStrategyBase, self).reset_common_infra(common_infra)
        if common_infra is not None:
            if "trade_exchange" in common_infra:
                self.trade_exchange = common_infra.get("trade_exchange")

    def reset(self, rely_trade_decision=None, **kwargs):
        """
        Parameters
        ----------
        rely_trade_decision : object, optional
        common_infra : None, optional
            common infrastructure for backtesting, by default None
            - It should include `trade_account`, used to get position
            - It should include `trade_exchange`, used to provide market info
        """
        super(SBBStrategyBase, self).reset(rely_trade_decision=rely_trade_decision, **kwargs)
        if rely_trade_decision is not None:
            self.trade_trend = {}
            self.trade_amount = {}
            # init the trade amount of order and  predicted trade trend
            for order in rely_trade_decision:
                self.trade_trend[(order.stock_id, order.direction)] = self.TREND_MID
                self.trade_amount[(order.stock_id, order.direction)] = order.amount

    def _pred_price_trend(self, stock_id, pred_start_time=None, pred_end_time=None):
        raise NotImplementedError("pred_price_trend method is not implemented!")

    def generate_trade_decision(self, execute_state):

        # update the order amount
        trade_info = execute_state
        for order, _, _, _ in trade_info:
            self.trade_amount[(order.stock_id, order.direction)] -= order.deal_amount
        trade_index = self.trade_calendar.get_trade_index()
        trade_len = self.trade_calendar.get_trade_len()
        trade_start_time, trade_end_time = self.trade_calendar.get_calendar_time(trade_index)
        pred_start_time, pred_end_time = self.trade_calendar.get_calendar_time(trade_index, shift=1)
        order_list = []
        # for each order in in self.rely_trade_decision
        for order in self.rely_trade_decision:
            # predict the price trend
            if trade_index % 2 == 1:
                _pred_trend = self._pred_price_trend(order.stock_id, pred_start_time, pred_end_time)
            else:
                _pred_trend = self.trade_trend[(order.stock_id, order.direction)]
            # if not tradable, continue
            if not self.trade_exchange.is_stock_tradable(
                stock_id=order.stock_id, start_time=trade_start_time, end_time=trade_end_time
            ):
                if trade_index % 2 == 1:
                    self.trade_trend[(order.stock_id, order.direction)] = _pred_trend
                continue
            # get amount of one trade unit
            _amount_trade_unit = self.trade_exchange.get_amount_of_trade_unit(order.factor)
            if _pred_trend == self.TREND_MID:
                _order_amount = None
                # considering trade unit
                if _amount_trade_unit is None:
                    # split the order equally
                    _order_amount = self.trade_amount[(order.stock_id, order.direction)] / (trade_len - trade_index + 1)
                # without considering trade unit
                elif self.trade_amount[(order.stock_id, order.direction)] >= _amount_trade_unit:
                    # cal how many trade unit
                    trade_unit_cnt = int(self.trade_amount[(order.stock_id, order.direction)] // _amount_trade_unit)
                    # split the order equally
                    # floor((trade_unit_cnt + trade_len - trade_index) / (trade_len - trade_index + 1)) == ceil(trade_unit_cnt / (trade_len - trade_index + 1))
                    _order_amount = (
                        (trade_unit_cnt + trade_len - trade_index) // (trade_len - trade_index + 1) * _amount_trade_unit
                    )
                if order.direction == order.SELL:
                    # sell all amount at last
                    if self.trade_amount[(order.stock_id, order.direction)] > 1e-5 and (
                        _order_amount is None or trade_index == trade_len
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
                    # N trade day last, split the order into N + 1 parts, and trade 2 parts
                    _order_amount = (
                        2 * self.trade_amount[(order.stock_id, order.direction)] / (trade_len - trade_index + 2)
                    )
                # without considering trade unit
                elif self.trade_amount[(order.stock_id, order.direction)] >= _amount_trade_unit:
                    # cal how many trade unit
                    trade_unit_cnt = int(self.trade_amount[(order.stock_id, order.direction)] // _amount_trade_unit)
                    # N trade day last, split the order into N + 1 parts, and trade 2 parts
                    _order_amount = (
                        (trade_unit_cnt + trade_len - trade_index + 1)
                        // (trade_len - trade_index + 2)
                        * 2
                        * _amount_trade_unit
                    )
                if order.direction == order.SELL:
                    # sell all amount at last
                    if self.trade_amount[(order.stock_id, order.direction)] >= 1e-5 and (
                        _order_amount is None or trade_index == trade_len
                    ):
                        _order_amount = self.trade_amount[(order.stock_id, order.direction)]

                if _order_amount:
                    _order_amount = min(_order_amount, self.trade_amount[(order.stock_id, order.direction)])
                    if trade_index % 2 == 1:
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

            if trade_index % 2 == 1:
                self.trade_trend[(order.stock_id, order.direction)] = _pred_trend

        return order_list


class SBBStrategyEMA(SBBStrategyBase):
    """
    (S)elect the (B)etter one among every two adjacent trading (B)ars to sell or buy with (EMA) signal.
    """

    def __init__(
        self,
        rely_trade_decision=[],
        instruments="csi300",
        freq="day",
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
        super(SBBStrategyEMA, self).__init__(rely_trade_decision, level_infra, common_infra, **kwargs)

    def _reset_signal(self):
        trade_len = self.trade_calendar.get_trade_len()
        fields = ["EMA($close, 10)-EMA($close, 20)"]
        signal_start_time, _ = self.trade_calendar.get_calendar_time(trade_index=1, shift=1)
        _, signal_end_time = self.trade_calendar.get_calendar_time(trade_index=trade_len, shift=1)
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
        - After reset the trade_calendar, the signal will be changed
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
