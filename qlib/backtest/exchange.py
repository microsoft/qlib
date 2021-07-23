# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from qlib.backtest.position import Position
import random
import logging
from typing import List, Tuple, Union, Callable, Iterable

import numpy as np
import pandas as pd

from ..data.data import D
from ..data.dataset.utils import get_level_index
from ..config import C, REG_CN
from ..utils.resam import resam_ts_data, ts_data_last
from ..log import get_module_logger
from .order import Order, OrderDir, OrderHelper
from .high_performance_ds import PandasQuote


class Exchange:
    def __init__(
        self,
        freq="day",
        start_time=None,
        end_time=None,
        codes="all",
        deal_price: Union[str, Tuple[str], List[str]] = None,
        subscribe_fields=[],
        limit_threshold: Union[Tuple[str, str], float, None] = None,
        volume_threshold=None,
        open_cost=0.0015,
        close_cost=0.0025,
        min_cost=5,
        extra_quote=None,
        quote_cls=PandasQuote,
        **kwargs,
    ):
        """__init__

        :param freq:             frequency of data
        :param start_time:       closed start time for backtest
        :param end_time:         closed end time for backtest
        :param codes:            list stock_id list or a string of instruments(i.e. all, csi500, sse50)

        :param deal_price:      Union[str, Tuple[str, str], List[str]]
                                The `deal_price` supports following two types of input
                                - <deal_price> : str
                                - (<buy_price>, <sell_price>): Tuple[str] or List[str]

                                <deal_price>, <buy_price> or <sell_price> := <price>
                                <price> := str
                                - for example '$close', '$open', '$vwap' ("close" is OK. `Exchange` will help to prepend
                                  "$" to the expression)

        :param subscribe_fields: list, subscribe fields. This expressions will be added to the query and `self.quote`.
                                 It is useful when users want more fields to be queried

        :param limit_threshold: Union[Tuple[str, str], float, None]
                                1) `None`: no limitation
                                2) float, 0.1 for example, default None
                                3) Tuple[str, str]: (<the expression for buying stock limitation>,
                                                     <the expression for sell stock limitation>)
                                                    `False` value indicates the stock is tradable
                                                    `True` value indicates the stock is limited and not tradable
        :param volume_threshold:  float, 0.1 for example, default None
        :param open_cost:        cost rate for open, default 0.0015
        :param close_cost:       cost rate for close, default 0.0025
        :param trade_unit:       trade unit, 100 for China A market.
                                 None for disable trade unit.
                                 **NOTE**: `trade_unit` is included in the `kwargs`. It is necessary because we must
                                 distinguish `not set` and `disable trade_unit`

        :param min_cost:         min cost, default 5
        :param extra_quote:     pandas, dataframe consists of
                                    columns: like ['$vwap', '$close', '$volume', '$factor', 'limit_sell', 'limit_buy'].
                                            The limit indicates that the etf is tradable on a specific day.
                                            Necessary fields:
                                                $close is for calculating the total value at end of each day.
                                            Optional fields:
                                                $volume is only necessary when we limit the trade amount or caculate PA(vwap) indicator
                                                $vwap is only necessary when we use the $vwap price as the deal price
                                                $factor is for rounding to the trading unit
                                                limit_sell will be set to False by default(False indicates we can sell this
                                                target on this day).
                                                limit_buy will be set to False by default(False indicates we can buy this
                                                target on this day).
                                    index: MultipleIndex(instrument, pd.Datetime)
        """
        self.freq = freq
        self.start_time = start_time
        self.end_time = end_time

        self.trade_unit = kwargs.pop("trade_unit", C.trade_unit)
        if len(kwargs) > 0:
            raise ValueError(f"Get Unexpected arguments {kwargs}")

        if limit_threshold is None:
            limit_threshold = C.limit_threshold
        if deal_price is None:
            deal_price = C.deal_price

        self.logger = get_module_logger("online operator", level=logging.INFO)

        # TODO: the quote, trade_dates, codes are not necessray.
        # It is just for performance consideration.
        self.limit_type = self._get_limit_type(limit_threshold)
        if limit_threshold is None:
            if C.region == REG_CN:
                self.logger.warning(f"limit_threshold not set. The stocks hit the limit may be bought/sold")
        elif self.limit_type == self.LT_FLT and abs(limit_threshold) > 0.1:
            if C.region == REG_CN:
                self.logger.warning(f"limit_threshold may not be set to a reasonable value")

        if isinstance(deal_price, str):
            if deal_price[0] != "$":
                deal_price = "$" + deal_price
            self.buy_price = self.sell_price = deal_price
        elif isinstance(deal_price, (tuple, list)):
            self.buy_price, self.sell_price = deal_price
        else:
            raise NotImplementedError(f"This type of input is not supported")

        if isinstance(codes, str):
            codes = D.instruments(codes)
        self.codes = codes
        # Necessary fields
        # $close is for calculating the total value at end of each day.
        # $factor is for rounding to the trading unit
        # $change is for calculating the limit of the stock

        necessary_fields = {self.buy_price, self.sell_price, "$close", "$change", "$factor", "$volume"}
        if self.limit_type == self.LT_TP_EXP:
            for exp in limit_threshold:
                necessary_fields.add(exp)
        all_fields = list(necessary_fields | set(subscribe_fields))

        self.all_fields = all_fields
        self.open_cost = open_cost
        self.close_cost = close_cost
        self.min_cost = min_cost
        self.limit_threshold: Union[Tuple[str, str], float, None] = limit_threshold
        self.volume_threshold = volume_threshold
        self.extra_quote = extra_quote
        self.get_quote_from_qlib()

        # init quote by quote_df
        self.quote_cls = quote_cls
        self.quote = self.quote_cls(self.quote_df)

    def get_quote_from_qlib(self):
        # get stock data from qlib
        if len(self.codes) == 0:
            self.codes = D.instruments()
        self.quote_df = D.features(
            self.codes, self.all_fields, self.start_time, self.end_time, freq=self.freq, disk_cache=True
        ).dropna(subset=["$close"])
        self.quote_df.columns = self.all_fields

        # check buy_price data and sell_price data
        for attr in "buy_price", "sell_price":
            pstr = getattr(self, attr)  # price string
            if self.quote_df[pstr].isna().any():
                self.logger.warning("{} field data contains nan.".format(pstr))

        # update trade_w_adj_price
        if self.quote_df["$factor"].isna().any():
            # The 'factor.day.bin' file not exists, and `factor` field contains `nan`
            # Use adjusted price
            self.trade_w_adj_price = True
            self.logger.warning("factor.day.bin file not exists or factor contains `nan`. Order using adjusted_price.")
            if self.trade_unit is not None:
                self.logger.warning(f"trade unit {self.trade_unit} is not supported in adjusted_price mode.")
        else:
            # The `factor.day.bin` file exists and all data `close` and `factor` are not `nan`
            # Use normal price
            self.trade_w_adj_price = False
        # update limit
        self._update_limit(self.limit_threshold)

        # concat extra_quote
        if self.extra_quote is not None:
            # process extra_quote
            if "$close" not in self.extra_quote:
                raise ValueError("$close is necessray in extra_quote")
            for attr in "buy_price", "sell_price":
                pstr = getattr(self, attr)  # price string
                if pstr not in self.extra_quote.columns:
                    self.extra_quote[pstr] = self.extra_quote["$close"]
                    self.logger.warning(f"No {pstr} set for extra_quote. Use $close as {pstr}.")
            if "$factor" not in self.extra_quote.columns:
                self.extra_quote["$factor"] = 1.0
                self.logger.warning("No $factor set for extra_quote. Use 1.0 as $factor.")
            if "limit_sell" not in self.extra_quote.columns:
                self.extra_quote["limit_sell"] = False
                self.logger.warning("No limit_sell set for extra_quote. All stock will be able to be sold.")
            if "limit_buy" not in self.extra_quote.columns:
                self.extra_quote["limit_buy"] = False
                self.logger.warning("No limit_buy set for extra_quote. All stock will be able to be bought.")
            assert set(self.extra_quote.columns) == set(self.quote_df.columns) - {"$change"}
            self.quote_df = pd.concat([self.quote_df, extra_quote], sort=False, axis=0)

    LT_TP_EXP = "(exp)"  # Tuple[str, str]
    LT_FLT = "float"  # float
    LT_NONE = "none"  # none

    def _get_limit_type(self, limit_threshold):
        """get limit type"""
        if isinstance(limit_threshold, Tuple):
            return self.LT_TP_EXP
        elif isinstance(limit_threshold, float):
            return self.LT_FLT
        elif limit_threshold is None:
            return self.LT_NONE
        else:
            raise NotImplementedError(f"This type of `limit_threshold` is not supported")

    def _update_limit(self, limit_threshold):
        # check limit_threshold
        limit_type = self._get_limit_type(limit_threshold)
        if limit_type == self.LT_NONE:
            self.quote_df["limit_buy"] = False
            self.quote_df["limit_sell"] = False
        elif limit_type == self.LT_TP_EXP:
            # set limit
            self.quote_df["limit_buy"] = self.quote_df[limit_threshold[0]]
            self.quote_df["limit_sell"] = self.quote_df[limit_threshold[1]]
        elif limit_type == self.LT_FLT:
            self.quote_df["limit_buy"] = self.quote_df["$change"].ge(limit_threshold)
            self.quote_df["limit_sell"] = self.quote_df["$change"].le(-limit_threshold)  # pylint: disable=E1130

    def check_stock_limit(self, stock_id, start_time, end_time, direction=None):
        """
        Parameters
        ----------
        direction : int, optional
            trade direction, by default None
            - if direction is None, check if tradable for buying and selling.
            - if direction == Order.BUY, check the if tradable for buying
            - if direction == Order.SELL, check the sell limit for selling.

        """
        if direction is None:
            buy_limit = self.quote.get_data(stock_id, start_time, end_time, fields="limit_buy", method="all")
            sell_limit = self.quote.get_data(stock_id, start_time, end_time, fields="limit_sell", method="all")
            return buy_limit or sell_limit
        elif direction == Order.BUY:
            return self.quote.get_data(stock_id, start_time, end_time, fields="limit_buy", method="all")
        elif direction == Order.SELL:
            return self.quote.get_data(stock_id, start_time, end_time, fields="limit_sell", method="all")
        else:
            raise ValueError(f"direction {direction} is not supported!")

    def check_stock_suspended(self, stock_id, start_time, end_time):
        # is suspended
        if stock_id in self.quote.get_all_stock():
            return self.quote.get_data(stock_id, start_time, end_time) is None
        else:
            return True

    def is_stock_tradable(self, stock_id, start_time, end_time, direction=None):
        # check if stock can be traded
        # same as check in check_order
        if self.check_stock_suspended(stock_id, start_time, end_time) or self.check_stock_limit(
            stock_id, start_time, end_time, direction
        ):
            return False
        else:
            return True

    def check_order(self, order):
        # check limit and suspended
        if self.check_stock_suspended(order.stock_id, order.start_time, order.end_time) or self.check_stock_limit(
            order.stock_id, order.start_time, order.end_time, order.direction
        ):
            return False
        else:
            return True

    def deal_order(self, order, trade_account=None, position=None):
        """
        Deal order when the actual transaction

        the results section in `Order` will be changed.

        :param order:  Deal the order.
        :param trade_account: Trade account to be updated after dealing the order.
        :param position: position to be updated after dealing the order.
        :return: trade_val, trade_cost, trade_price
        """
        # need to check order first
        # TODO: check the order unit limit in the exchange!!!!
        # The order limit is related to the adj factor and the cur_amount.
        # factor = self.quote[(order.stock_id, order.trade_date)]['$factor']
        # cur_amount = trade_account.current.get_stock_amount(order.stock_id)
        if self.check_order(order) is False:
            raise AttributeError("need to check order first")
        if trade_account is not None and position is not None:
            raise ValueError("trade_account and position can only choose one")

        trade_price = self.get_deal_price(order.stock_id, order.start_time, order.end_time, order.direction)
        # NOTE: order will be changed in this function
        trade_val, trade_cost = self._calc_trade_info_by_order(
            order, trade_account.current if trade_account else position
        )
        # update account
        if order.deal_amount > 1e-5:
            # If the order can only be deal 0 aomount. Nothing to be updated
            # Otherwise, it will result some stock with 0 amount in the position
            if trade_account:
                trade_account.update_order(order=order, trade_val=trade_val, cost=trade_cost, trade_price=trade_price)
            elif position:
                position.update_order(order=order, trade_val=trade_val, cost=trade_cost, trade_price=trade_price)

        return trade_val, trade_cost, trade_price

    def get_quote_info(self, stock_id, start_time, end_time, method=ts_data_last):
        return self.quote.get_data(stock_id, start_time, end_time, method=method)

    def get_close(self, stock_id, start_time, end_time, method=ts_data_last):
        return self.quote.get_data(stock_id, start_time, end_time, fields="$close", method=method)

    def get_volume(self, stock_id, start_time, end_time, method="sum"):
        return self.quote.get_data(stock_id, start_time, end_time, fields="$volume", method=method)

    def get_deal_price(self, stock_id, start_time, end_time, direction: OrderDir, method=ts_data_last):
        if direction == OrderDir.SELL:
            pstr = self.sell_price
        elif direction == OrderDir.BUY:
            pstr = self.buy_price
        else:
            raise NotImplementedError(f"This type of input is not supported")
        deal_price = self.quote.get_data(stock_id, start_time, end_time, fields=pstr, method=method)
        if method is not None and (np.isclose(deal_price, 0.0) or np.isnan(deal_price)):
            self.logger.warning(f"(stock_id:{stock_id}, trade_time:{(start_time, end_time)}, {pstr}): {deal_price}!!!")
            self.logger.warning(f"setting deal_price to close price")
            deal_price = self.get_close(stock_id, start_time, end_time, method)
        return deal_price

    def get_factor(self, stock_id, start_time, end_time) -> Union[float, None]:
        """
        Returns
        -------
        Union[float, None]:
            `None`: if the stock is suspended `None` may be returned
            `float`: return factor if the factor exists
        """
        assert (start_time is not None and end_time is not None, "the time range must be given")
        if stock_id not in self.quote.get_all_stock():
            return None
        return self.quote.get_data(stock_id, start_time, end_time, fields="$factor", method=ts_data_last)

    def generate_amount_position_from_weight_position(
        self, weight_position, cash, start_time, end_time, direction=OrderDir.BUY
    ):
        """
        The generate the target position according to the weight and the cash.
        NOTE: All the cash will assigned to the tadable stock.

        Parameter:
        weight_position : dict {stock_id : weight}; allocate cash by weight_position
            among then, weight must be in this range: 0 < weight < 1
        cash : cash
        start_time : the start time point of the step
        end_time : the end time point of the step
        direction : the direction of the deal price for estimating the amount
                    # NOTE: this function is used for calculating target position. So the default direction is buy
        """

        # calculate the total weight of tradable value
        tradable_weight = 0.0
        for stock_id in weight_position:
            if self.is_stock_tradable(stock_id=stock_id, start_time=start_time, end_time=end_time):
                # weight_position must be greater than 0 and less than 1
                if weight_position[stock_id] < 0 or weight_position[stock_id] > 1:
                    raise ValueError(
                        "weight_position is {}, "
                        "weight_position is not in the range of (0, 1).".format(weight_position[stock_id])
                    )
                tradable_weight += weight_position[stock_id]

        if tradable_weight - 1.0 >= 1e-5:
            raise ValueError("tradable_weight is {}, can not greater than 1.".format(tradable_weight))

        amount_dict = {}
        for stock_id in weight_position:
            if weight_position[stock_id] > 0.0 and self.is_stock_tradable(
                stock_id=stock_id, start_time=start_time, end_time=end_time
            ):
                amount_dict[stock_id] = (
                    cash
                    * weight_position[stock_id]
                    / tradable_weight
                    // self.get_deal_price(
                        stock_id=stock_id, start_time=start_time, end_time=end_time, direction=direction
                    )
                )
        return amount_dict

    def get_real_deal_amount(self, current_amount, target_amount, factor):
        """
        Calculate the real adjust deal amount when considering the trading unit

        :param current_amount:
        :param target_amount:
        :param factor:
        :return  real_deal_amount;  Positive deal_amount indicates buying more stock.
        """
        if current_amount == target_amount:
            return 0
        elif current_amount < target_amount:
            deal_amount = target_amount - current_amount
            deal_amount = self.round_amount_by_trade_unit(deal_amount, factor)
            return deal_amount
        else:
            if target_amount == 0:
                return -current_amount
            else:
                deal_amount = current_amount - target_amount
                deal_amount = self.round_amount_by_trade_unit(deal_amount, factor)
                return -deal_amount

    def generate_order_for_target_amount_position(self, target_position, current_position, start_time, end_time):
        """
        Note: some future information is used in this function

        Parameter:
        target_position : dict { stock_id : amount }
        current_postion : dict { stock_id : amount}
        trade_unit : trade_unit
        down sample : for amount 321 and trade_unit 100, deal_amount is 300
        deal order on trade_date
        """
        # split buy and sell for further use
        buy_order_list = []
        sell_order_list = []
        # three parts: kept stock_id, dropped stock_id, new stock_id
        # handle kept stock_id

        # because the order of the set is not fixed, the trading order of the stock is different, so that the backtest results of the same parameter are different;
        # so here we sort stock_id, and then randomly shuffle the order of stock_id
        # because the same random seed is used, the final stock_id order is fixed
        sorted_ids = sorted(set(list(current_position.keys()) + list(target_position.keys())))
        random.seed(0)
        random.shuffle(sorted_ids)
        for stock_id in sorted_ids:

            # Do not generate order for the nontradable stocks
            if not self.is_stock_tradable(stock_id=stock_id, start_time=start_time, end_time=end_time):
                continue

            target_amount = target_position.get(stock_id, 0)
            current_amount = current_position.get(stock_id, 0)
            factor = self.get_factor(stock_id, start_time=start_time, end_time=end_time)

            deal_amount = self.get_real_deal_amount(current_amount, target_amount, factor)
            if deal_amount == 0:
                continue
            elif deal_amount > 0:
                # buy stock
                buy_order_list.append(
                    Order(
                        stock_id=stock_id,
                        amount=deal_amount,
                        direction=Order.BUY,
                        start_time=start_time,
                        end_time=end_time,
                        factor=factor,
                    )
                )
            else:
                # sell stock
                sell_order_list.append(
                    Order(
                        stock_id=stock_id,
                        amount=abs(deal_amount),
                        direction=Order.SELL,
                        start_time=start_time,
                        end_time=end_time,
                        factor=factor,
                    )
                )
        # return order_list : buy + sell
        return sell_order_list + buy_order_list

    def calculate_amount_position_value(
        self, amount_dict, start_time, end_time, only_tradable=False, direction=OrderDir.SELL
    ):
        """Parameter
        position : Position()
        amount_dict : {stock_id : amount}
        direction : the direction of the deal price for estimating the amount
                    # NOTE:
                    This function is used for calculating current position value.
                    So the default direction is sell.
        """
        value = 0
        for stock_id in amount_dict:
            if (
                self.check_stock_suspended(stock_id=stock_id, start_time=start_time, end_time=end_time) is False
                and self.check_stock_limit(stock_id=stock_id, start_time=start_time, end_time=end_time) is False
            ):
                value += (
                    self.get_deal_price(
                        stock_id=stock_id, start_time=start_time, end_time=end_time, direction=direction
                    )
                    * amount_dict[stock_id]
                )
        return value

    def _get_factor_or_raise_erorr(self, factor: float = None, stock_id: str = None, start_time=None, end_time=None):
        """Please refer to the docs of get_amount_of_trade_unit"""
        if factor is None:
            if stock_id is not None and start_time is not None and end_time is not None:
                factor = self.get_factor(stock_id=stock_id, start_time=start_time, end_time=end_time)
            else:
                raise ValueError(f"`factor` and (`stock_id`, `start_time`, `end_time`) can't both be None")
        return factor

    def get_amount_of_trade_unit(self, factor: float = None, stock_id: str = None, start_time=None, end_time=None):
        """
        get the trade unit of amount based on **factor**

        the factor can be given directly or calculated in given time range and stock id.
        `factor` has higher priority than `stock_id`, `start_time` and `end_time`

        Parameters
        ----------
        factor : float
            the adjusted factor
        stock_id : str
            the id of the stock
        start_time :
            the start time of trading range
        end_time :
            the end time of trading range
        """
        if not self.trade_w_adj_price and self.trade_unit is not None:
            factor = self._get_factor_or_raise_erorr(
                factor=factor, stock_id=stock_id, start_time=start_time, end_time=end_time
            )
            return self.trade_unit / factor
        else:
            return None

    def round_amount_by_trade_unit(
        self, deal_amount, factor: float = None, stock_id: str = None, start_time=None, end_time=None
    ):
        """Parameter
        Please refer to the docs of get_amount_of_trade_unit

        deal_amount : float, adjusted amount
        factor : float, adjusted factor
        return : float, real amount
        """
        if not self.trade_w_adj_price and self.trade_unit is not None:
            # the minimal amount is 1. Add 0.1 for solving precision problem.
            factor = self._get_factor_or_raise_erorr(
                factor=factor, stock_id=stock_id, start_time=start_time, end_time=end_time
            )
            return (deal_amount * factor + 0.1) // self.trade_unit * self.trade_unit / factor
        return deal_amount

    def _get_amount_by_volume(self, stock_id, trade_start_time, trade_end_time, deal_amount):
        if self.volume_threshold is not None:
            tradable_amount = self.get_volume(stock_id, trade_start_time, trade_end_time) * self.volume_threshold
            return max(min(tradable_amount, deal_amount), 0)
        else:
            return deal_amount

    def _calc_trade_info_by_order(self, order, position: Position):
        """
        Calculation of trade info

        **NOTE**: Order will be changed in this function

        :param order:
        :param position: Position
        :return: trade_val, trade_cost
        """

        trade_price = self.get_deal_price(order.stock_id, order.start_time, order.end_time, direction=order.direction)
        order.factor = self.get_factor(order.stock_id, order.start_time, order.end_time)
        if order.direction == Order.SELL:
            # sell
            if position is not None:
                current_amount = (
                    position.get_stock_amount(order.stock_id) if position.check_stock(order.stock_id) else 0
                )
                if np.isclose(order.amount, current_amount):
                    # when selling last stock. The amount don't need rounding
                    order.deal_amount = order.amount
                elif order.amount > current_amount:
                    order.deal_amount = self.round_amount_by_trade_unit(current_amount, order.factor)
                else:
                    order.deal_amount = self.round_amount_by_trade_unit(order.amount, order.factor)
            else:
                # TODO: We don't know current position.
                #  We choose to sell all
                order.deal_amount = order.amount

            order.deal_amount = self._get_amount_by_volume(
                order.stock_id, order.start_time, order.end_time, order.deal_amount
            )
            trade_val = order.deal_amount * trade_price
            trade_cost = max(trade_val * self.close_cost, self.min_cost)
        elif order.direction == Order.BUY:
            # buy
            if position is not None:
                cash = position.get_cash()
                trade_val = order.amount * trade_price
                if cash < trade_val * (1 + self.open_cost):
                    # The money is not enough
                    order.deal_amount = self.round_amount_by_trade_unit(
                        cash / (1 + self.open_cost) / trade_price, order.factor
                    )
                else:
                    # THe money is enough
                    order.deal_amount = self.round_amount_by_trade_unit(order.amount, order.factor)
            else:
                # Unknown amount of money. Just round the amount
                order.deal_amount = self.round_amount_by_trade_unit(order.amount, order.factor)

            order.deal_amount = self._get_amount_by_volume(
                order.stock_id, order.start_time, order.end_time, order.deal_amount
            )
            trade_val = order.deal_amount * trade_price
            trade_cost = trade_val * self.open_cost
        else:
            raise NotImplementedError("order type {} error".format(order.type))

        return trade_val, trade_cost

    def get_order_helper(self) -> OrderHelper:
        if not hasattr(self, "_order_helper"):
            # cache to avoid recreate the same instance
            self._order_helper = OrderHelper(self)
        return self._order_helper
