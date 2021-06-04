# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import random
import logging

import numpy as np
import pandas as pd

from ..data.data import D
from ..data.dataset.utils import get_level_index
from ..config import C, REG_CN
from ..utils.resam import resam_ts_data
from ..log import get_module_logger
from .order import Order


class Exchange:
    def __init__(
        self,
        freq="day",
        start_time=None,
        end_time=None,
        codes="all",
        deal_price=None,
        subscribe_fields=[],
        limit_threshold=None,
        volume_threshold=None,
        open_cost=0.0015,
        close_cost=0.0025,
        trade_unit=None,
        min_cost=5,
        extra_quote=None,
    ):
        """__init__

        :param freq:             frequency of data
        :param start_time:       closed start time for backtest
        :param end_time:         closed end time for backtest
        :param codes:            list stock_id list or a string of instruments(i.e. all, csi500, sse50)
        :param deal_price:       str, 'close', 'open', 'vwap'
        :param subscribe_fields: list, subscribe fields
        :param limit_threshold:  float, 0.1 for example, default None
        :param volume_threshold:  float, 0.1 for example, default None
        :param open_cost:        cost rate for open, default 0.0015
        :param close_cost:       cost rate for close, default 0.0025
        :param trade_unit:       trade unit, 100 for China A market
        :param min_cost:         min cost, default 5
        :param extra_quote:     pandas, dataframe consists of
                                    columns: like ['$vwap', '$close', '$factor', 'limit'].
                                            The limit indicates that the etf is tradable on a specific day.
                                            Necessary fields:
                                                $close is for calculating the total value at end of each day.
                                            Optional fields:
                                                $vwap is only necessary when we use the $vwap price as the deal price
                                                $factor is for rounding to the trading unit
                                                limit will be set to False by default(False indicates we can buy this
                                                target on this day).
                                    index: MultipleIndex(instrument, pd.Datetime)
        """
        self.freq = freq
        self.start_time = start_time
        self.end_time = end_time

        if trade_unit is None:
            trade_unit = C.trade_unit
        if limit_threshold is None:
            limit_threshold = C.limit_threshold
        if deal_price is None:
            deal_price = C.deal_price

        self.logger = get_module_logger("online operator", level=logging.INFO)

        self.trade_unit = trade_unit
        # TODO: the quote, trade_dates, codes are not necessray.
        # It is just for performance consideration.
        if limit_threshold is None:
            if C.region == REG_CN:
                self.logger.warning(f"limit_threshold not set. The stocks hit the limit may be bought/sold")
        elif abs(limit_threshold) > 0.1:
            if C.region == REG_CN:
                self.logger.warning(f"limit_threshold may not be set to a reasonable value")

        if deal_price[0] != "$":
            self.deal_price = "$" + deal_price
        else:
            self.deal_price = deal_price
        if isinstance(codes, str):
            codes = D.instruments(codes)
        self.codes = codes
        # Necessary fields
        # $close is for calculating the total value at end of each day.
        # $factor is for rounding to the trading unit
        # $change is for calculating the limit of the stock

        necessary_fields = {self.deal_price, "$close", "$change", "$factor", "$volume"}
        subscribe_fields = list(necessary_fields | set(subscribe_fields))
        all_fields = list(necessary_fields | set(subscribe_fields))
        self.all_fields = all_fields
        self.open_cost = open_cost
        self.close_cost = close_cost
        self.min_cost = min_cost
        self.limit_threshold = limit_threshold
        self.volume_threshold = volume_threshold
        self.extra_quote = extra_quote
        self.set_quote(codes, start_time, end_time)

    def set_quote(self, codes, start_time, end_time):
        if len(codes) == 0:
            codes = D.instruments()
        self.quote = D.features(codes, self.all_fields, start_time, end_time, freq=self.freq, disk_cache=True).dropna(
            subset=["$close"]
        )
        self.quote.columns = self.all_fields

        if self.quote[self.deal_price].isna().any():
            self.logger.warning("{} field data contains nan.".format(self.deal_price))

        if self.quote["$factor"].isna().any():
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
        # check limit_threshold
        if self.limit_threshold is None:
            self.quote["limit_buy"] = False
            self.quote["limit_sell"] = False
        else:
            # set limit
            self._update_limit(buy_limit=self.limit_threshold, sell_limit=self.limit_threshold)

        quote_df = self.quote
        if self.extra_quote is not None:
            # process extra_quote
            if "$close" not in self.extra_quote:
                raise ValueError("$close is necessray in extra_quote")
            if self.deal_price not in self.extra_quote.columns:
                self.extra_quote[self.deal_price] = self.extra_quote["$close"]
                self.logger.warning("No deal_price set for extra_quote. Use $close as deal_price.")
            if "$factor" not in self.extra_quote.columns:
                self.extra_quote["$factor"] = 1.0
                self.logger.warning("No $factor set for extra_quote. Use 1.0 as $factor.")
            if "limit_sell" not in self.extra_quote.columns:
                self.extra_quote["limit_sell"] = False
                self.logger.warning("No limit_sell set for extra_quote. All stock will be able to be sold.")
            if "limit_buy" not in self.extra_quote.columns:
                self.extra_quote["limit_buy"] = False
                self.logger.warning("No limit_buy set for extra_quote. All stock will be able to be bought.")

            assert set(self.extra_quote.columns) == set(quote_df.columns) - {"$change"}
            quote_df = pd.concat([quote_df, self.extra_quote], sort=False, axis=0)

        # update quote: pd.DataFrame to dict, for search use
        if get_level_index(quote_df, level="datetime") == 1:
            quote_df = quote_df.swaplevel().sort_index()

        quote_dict = {}
        for stock_id, stock_val in quote_df.groupby(level="instrument"):
            quote_dict[stock_id] = stock_val

        self.quote = quote_dict

    def _update_limit(self, buy_limit, sell_limit):
        self.quote["limit_buy"] = ~self.quote["$change"].lt(buy_limit)
        self.quote["limit_sell"] = ~self.quote["$change"].gt(-sell_limit)

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
            buy_limit = resam_ts_data(self.quote[stock_id]["limit_buy"], start_time, end_time, method="all").iloc[0]
            sell_limit = resam_ts_data(self.quote[stock_id]["limit_sell"], start_time, end_time, method="all").iloc[0]
            return buy_limit or sell_limit
        elif direction == Order.BUY:
            return resam_ts_data(self.quote[stock_id]["limit_buy"], start_time, end_time, method="all").iloc[0]
        elif direction == Order.SELL:
            return resam_ts_data(self.quote[stock_id]["limit_sell"], start_time, end_time, method="all").iloc[0]
        else:
            raise ValueError(f"direction {direction} is not supported!")

    def check_stock_suspended(self, stock_id, start_time, end_time):
        # is suspended
        if stock_id in self.quote:
            return resam_ts_data(self.quote[stock_id], start_time, end_time, method=None) is None
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

        trade_price = self.get_deal_price(order.stock_id, order.start_time, order.end_time)
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

    def get_quote_info(self, stock_id, start_time, end_time):
        return resam_ts_data(self.quote[stock_id], start_time, end_time, method="last").iloc[0]

    def get_close(self, stock_id, start_time, end_time):
        return resam_ts_data(self.quote[stock_id]["$close"], start_time, end_time, method="last").iloc[0]

    def get_volume(self, stock_id, start_time, end_time):
        return resam_ts_data(self.quote[stock_id]["$volume"], start_time, end_time, method="sum").iloc[0]

    def get_deal_price(self, stock_id, start_time, end_time):
        deal_price = resam_ts_data(self.quote[stock_id][self.deal_price], start_time, end_time, method="last").iloc[0]
        if np.isclose(deal_price, 0.0) or np.isnan(deal_price):
            self.logger.warning(
                f"(stock_id:{stock_id}, trade_time:{(start_time, end_time)}, {self.deal_price}): {deal_price}!!!"
            )
            self.logger.warning(f"setting deal_price to close price")
            deal_price = self.get_close(stock_id, start_time, end_time)
        return deal_price

    def get_factor(self, stock_id, start_time, end_time):
        return resam_ts_data(self.quote[stock_id]["$factor"], start_time, end_time, method="last").iloc[0]

    def generate_amount_position_from_weight_position(self, weight_position, cash, start_time, end_time):
        """
        The generate the target position according to the weight and the cash.
        NOTE: All the cash will assigned to the tadable stock.

        Parameter:
        weight_position : dict {stock_id : weight}; allocate cash by weight_position
            among then, weight must be in this range: 0 < weight < 1
        cash : cash
        trade_date : trade date
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
                    // self.get_deal_price(stock_id=stock_id, start_time=start_time, end_time=end_time)
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
        """Parameter:
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

    def calculate_amount_position_value(self, amount_dict, start_time, end_time, only_tradable=False):
        """Parameter
        position : Position()
        amount_dict : {stock_id : amount}
        """
        value = 0
        for stock_id in amount_dict:
            if (
                self.check_stock_suspended(stock_id=stock_id, start_time=start_time, end_time=end_time) is False
                and self.check_stock_limit(stock_id=stock_id, start_time=start_time, end_time=end_time) is False
            ):
                value += (
                    self.get_deal_price(stock_id=stock_id, start_time=start_time, end_time=end_time)
                    * amount_dict[stock_id]
                )
        return value

    def get_amount_of_trade_unit(self, factor):
        if not self.trade_w_adj_price and self.trade_unit is not None:
            return self.trade_unit / factor
        else:
            return None

    def round_amount_by_trade_unit(self, deal_amount, factor):
        """Parameter
        deal_amount : float, adjusted amount
        factor : float, adjusted factor
        return : float, real amount
        """
        if not self.trade_w_adj_price and self.trade_unit is not None:
            # the minimal amount is 1. Add 0.1 for solving precision problem.
            return (deal_amount * factor + 0.1) // self.trade_unit * self.trade_unit / factor
        return deal_amount

    def _get_amount_by_volume(self, stock_id, trade_start_time, trade_end_time, deal_amount):
        if self.volume_threshold is not None:
            tradable_amount = self.get_volume(stock_id, trade_start_time, trade_end_time) * self.volume_threshold
            return max(min(tradable_amount, deal_amount), 0)
        else:
            return deal_amount

    def _calc_trade_info_by_order(self, order, position):
        """
        Calculation of trade info

        :param order:
        :param position: Position
        :return: trade_val, trade_cost
        """

        trade_price = self.get_deal_price(order.stock_id, order.start_time, order.end_time)
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
