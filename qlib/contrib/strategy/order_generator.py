# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
This order generator is for strategies based on WeightStrategyBase
"""
from ..backtest.position import Position
from ..backtest.exchange import Exchange
import pandas as pd
import copy


class OrderGenerator:
    def generate_order_list_from_target_weight_position(
        self,
        current: Position,
        trade_exchange: Exchange,
        target_weight_position: dict,
        risk_degree: float,
        pred_date: pd.Timestamp,
        trade_date: pd.Timestamp,
    ) -> list:
        """generate_order_list_from_target_weight_position

        :param current: The current position
        :type current: Position
        :param trade_exchange:
        :type trade_exchange: Exchange
        :param target_weight_position: {stock_id : weight}
        :type target_weight_position: dict
        :param risk_degree:
        :type risk_degree: float
        :param pred_date: the date the score is predicted
        :type pred_date: pd.Timestamp
        :param trade_date: the date the stock is traded
        :type trade_date: pd.Timestamp

        :rtype: list
        """
        raise NotImplementedError()


class OrderGenWInteract(OrderGenerator):
    """Order Generator With Interact"""

    def generate_order_list_from_target_weight_position(
        self,
        current: Position,
        trade_exchange: Exchange,
        target_weight_position: dict,
        risk_degree: float,
        pred_date: pd.Timestamp,
        trade_date: pd.Timestamp,
    ) -> list:
        """generate_order_list_from_target_weight_position

        No adjustment for for the nontradable share.
        All the tadable value is assigned to the tadable stock according to the weight.
        if interact == True, will use the price at trade date to generate order list
        else, will only use the price before the trade date to generate order list

        :param current:
        :type current: Position
        :param trade_exchange:
        :type trade_exchange: Exchange
        :param target_weight_position:
        :type target_weight_position: dict
        :param risk_degree:
        :type risk_degree: float
        :param pred_date:
        :type pred_date: pd.Timestamp
        :param trade_date:
        :type trade_date: pd.Timestamp

        :rtype: list
        """
        # calculate current_tradable_value
        current_amount_dict = current.get_stock_amount_dict()
        current_total_value = trade_exchange.calculate_amount_position_value(
            amount_dict=current_amount_dict, trade_date=trade_date, only_tradable=False
        )
        current_tradable_value = trade_exchange.calculate_amount_position_value(
            amount_dict=current_amount_dict, trade_date=trade_date, only_tradable=True
        )
        # add cash
        current_tradable_value += current.get_cash()

        reserved_cash = (1.0 - risk_degree) * (current_total_value + current.get_cash())
        current_tradable_value -= reserved_cash

        if current_tradable_value < 0:
            # if you sell all the tradable stock can not meet the reserved
            # value. Then just sell all the stocks
            target_amount_dict = copy.deepcopy(current_amount_dict.copy())
            for stock_id in list(target_amount_dict.keys()):
                if trade_exchange.is_stock_tradable(stock_id, trade_date):
                    del target_amount_dict[stock_id]
        else:
            # consider cost rate
            current_tradable_value /= 1 + max(trade_exchange.close_cost, trade_exchange.open_cost)

            # strategy 1 : generate amount_position by weight_position
            # Use API in Exchange()
            target_amount_dict = trade_exchange.generate_amount_position_from_weight_position(
                weight_position=target_weight_position,
                cash=current_tradable_value,
                trade_date=trade_date,
            )
        order_list = trade_exchange.generate_order_for_target_amount_position(
            target_position=target_amount_dict,
            current_position=current_amount_dict,
            trade_date=trade_date,
        )
        return order_list


class OrderGenWOInteract(OrderGenerator):
    """Order Generator Without Interact"""

    def generate_order_list_from_target_weight_position(
        self,
        current: Position,
        trade_exchange: Exchange,
        target_weight_position: dict,
        risk_degree: float,
        pred_date: pd.Timestamp,
        trade_date: pd.Timestamp,
    ) -> list:
        """generate_order_list_from_target_weight_position

        generate order list directly not using the information (e.g. whether can be traded, the accurate trade price) at trade date.
        In target weight position, generating order list need to know the price of objective stock in trade date, but we cannot get that
        value when do not interact with exchange, so we check the %close price at pred_date or price recorded in current position.

        :param current:
        :type current: Position
        :param trade_exchange:
        :type trade_exchange: Exchange
        :param target_weight_position:
        :type target_weight_position: dict
        :param risk_degree:
        :type risk_degree: float
        :param pred_date:
        :type pred_date: pd.Timestamp
        :param trade_date:
        :type trade_date: pd.Timestamp

        :rtype: list
        """
        risk_total_value = risk_degree * current.calculate_value()

        current_stock = current.get_stock_list()
        amount_dict = {}
        for stock_id in target_weight_position:
            # Current rule will ignore the stock that not hold and cannot be traded at predict date
            if trade_exchange.is_stock_tradable(stock_id=stock_id, trade_date=pred_date):
                amount_dict[stock_id] = (
                    risk_total_value * target_weight_position[stock_id] / trade_exchange.get_close(stock_id, pred_date)
                )
            elif stock_id in current_stock:
                amount_dict[stock_id] = (
                    risk_total_value * target_weight_position[stock_id] / current.get_stock_price(stock_id)
                )
            else:
                continue
        order_list = trade_exchange.generate_order_for_target_amount_position(
            target_position=amount_dict,
            current_position=current.get_stock_amount_dict(),
            trade_date=trade_date,
        )
        return order_list
