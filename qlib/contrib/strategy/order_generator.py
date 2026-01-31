# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
This order generator is for strategies based on WeightStrategyBase
"""
from ...backtest.position import Position
from ...backtest.exchange import Exchange

import pandas as pd
import copy


class OrderGenerator:
    """Base class for order generators used by ``WeightStrategyBase``.

    An order generator converts a target weight position (a mapping from
    stock_id to portfolio weight) into a concrete list of ``Order`` objects
    that can be executed by the ``Exchange``.

    Two built-in implementations are provided:

    * ``OrderGenWInteract`` – uses trade-date market information (prices,
      tradability) when building orders. It automatically **re-normalises**
      weights across *tradable* stocks so that the full allocatable capital
      is utilised. Use this when the executor can interact with the exchange
      at execution time.
    * ``OrderGenWOInteract`` – generates orders **without** accessing
      trade-date information. It relies on the prediction-date close price
      (or the price recorded in the current position) to estimate order
      amounts. This is the default used by ``WeightStrategyBase``.

    Subclass this and override
    ``generate_order_list_from_target_weight_position`` to implement custom
    order generation logic.
    """

    def generate_order_list_from_target_weight_position(
        self,
        current: Position,
        trade_exchange: Exchange,
        target_weight_position: dict,
        risk_degree: float,
        pred_start_time: pd.Timestamp,
        pred_end_time: pd.Timestamp,
        trade_start_time: pd.Timestamp,
        trade_end_time: pd.Timestamp,
    ) -> list:
        """Generate a list of orders from the target weight position.

        :param current: The current portfolio position.
        :type current: Position
        :param trade_exchange: The exchange instance providing market data,
            tradability checks, and order-rounding utilities.
        :type trade_exchange: Exchange
        :param target_weight_position: Mapping ``{stock_id: weight}`` where
            each weight is a float in ``(0, 1)`` representing the desired
            portfolio proportion for that stock.
        :type target_weight_position: dict
        :param risk_degree: Fraction of total portfolio value that may be
            allocated to risky assets (stocks). ``1.0`` means fully invested.
        :type risk_degree: float
        :param pred_start_time: Start of the prediction time window.
        :type pred_start_time: pd.Timestamp
        :param pred_end_time: End of the prediction time window.
        :type pred_end_time: pd.Timestamp
        :param trade_start_time: Start of the actual trading time window.
        :type trade_start_time: pd.Timestamp
        :param trade_end_time: End of the actual trading time window.
        :type trade_end_time: pd.Timestamp

        :rtype: list
        :returns: A list of ``Order`` objects.
        """
        raise NotImplementedError()


class OrderGenWInteract(OrderGenerator):
    """Order generator that uses trade-date market information.

    This generator **interacts** with the exchange at execution time to
    obtain accurate trade-date prices and tradability status. It
    re-normalises the target weights so that the full tradable capital is
    distributed only among stocks that are actually tradable on the trade
    date.

    Key behaviour:
    * Calls ``Exchange.generate_amount_position_from_weight_position`` which
      divides cash among tradable stocks proportionally to their weights,
      effectively **ignoring suspended or limited stocks** and
      redistributing their weight to the remaining tradable stocks.
    * This ensures full capital utilisation when some stocks become
      untradable between the prediction date and the trade date.

    See Also
    --------
    OrderGenWOInteract : Alternative that does **not** use trade-date data.
    """

    def generate_order_list_from_target_weight_position(
        self,
        current: Position,
        trade_exchange: Exchange,
        target_weight_position: dict,
        risk_degree: float,
        pred_start_time: pd.Timestamp,
        pred_end_time: pd.Timestamp,
        trade_start_time: pd.Timestamp,
        trade_end_time: pd.Timestamp,
    ) -> list:
        """Generate orders using trade-date prices and tradability data.

        The tradable portfolio value is computed from the current position
        valued at trade-date prices. Weights are then allocated only across
        stocks that are tradable on the trade date, so the full allocatable
        capital is utilised even when some target stocks are suspended.

        :param current: The current portfolio position.
        :type current: Position
        :param trade_exchange: Exchange providing trade-date market data.
        :type trade_exchange: Exchange
        :param target_weight_position: ``{stock_id: weight}`` mapping.
        :type target_weight_position: dict
        :param risk_degree: Fraction of portfolio allocated to stocks.
        :type risk_degree: float
        :param pred_start_time: Start of the prediction window.
        :type pred_start_time: pd.Timestamp
        :param pred_end_time: End of the prediction window.
        :type pred_end_time: pd.Timestamp
        :param trade_start_time: Start of the trading window.
        :type trade_start_time: pd.Timestamp
        :param trade_end_time: End of the trading window.
        :type trade_end_time: pd.Timestamp

        :rtype: list
        :returns: A list of ``Order`` objects.
        """
        if target_weight_position is None:
            return []

        # calculate current_tradable_value
        current_amount_dict = current.get_stock_amount_dict()

        current_total_value = trade_exchange.calculate_amount_position_value(
            amount_dict=current_amount_dict,
            start_time=trade_start_time,
            end_time=trade_end_time,
            only_tradable=False,
        )
        current_tradable_value = trade_exchange.calculate_amount_position_value(
            amount_dict=current_amount_dict,
            start_time=trade_start_time,
            end_time=trade_end_time,
            only_tradable=True,
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
                if trade_exchange.is_stock_tradable(stock_id, start_time=trade_start_time, end_time=trade_end_time):
                    del target_amount_dict[stock_id]
        else:
            # consider cost rate
            current_tradable_value /= 1 + max(trade_exchange.close_cost, trade_exchange.open_cost)

            # strategy 1 : generate amount_position by weight_position
            # Use API in Exchange()
            target_amount_dict = trade_exchange.generate_amount_position_from_weight_position(
                weight_position=target_weight_position,
                cash=current_tradable_value,
                start_time=trade_start_time,
                end_time=trade_end_time,
            )
        order_list = trade_exchange.generate_order_for_target_amount_position(
            target_position=target_amount_dict,
            current_position=current_amount_dict,
            start_time=trade_start_time,
            end_time=trade_end_time,
        )
        return order_list


class OrderGenWOInteract(OrderGenerator):
    """Order generator that does **not** use trade-date market information.

    This is the **default** order generator for ``WeightStrategyBase``.

    Because trade-date prices are unavailable at decision time, this
    generator estimates order amounts using:

    1. The **prediction-date close price** (``$close`` at ``pred_date``) for
       stocks that are tradable on both the prediction date and the trade
       date.
    2. The **price recorded in the current position** for stocks that are
       currently held but not tradable on the prediction date.

    Unlike ``OrderGenWInteract``, this generator does **not** re-normalise
    weights across tradable stocks. Stocks that are untradable on the trade
    date are simply skipped, which may result in less than full capital
    utilisation.

    See Also
    --------
    OrderGenWInteract : Alternative that re-normalises weights using
        trade-date data for full capital utilisation.
    """

    def generate_order_list_from_target_weight_position(
        self,
        current: Position,
        trade_exchange: Exchange,
        target_weight_position: dict,
        risk_degree: float,
        pred_start_time: pd.Timestamp,
        pred_end_time: pd.Timestamp,
        trade_start_time: pd.Timestamp,
        trade_end_time: pd.Timestamp,
    ) -> list:
        """Generate orders without accessing trade-date information.

        Order amounts are estimated from prediction-date close prices or
        prices recorded in the current position. Stocks that are untradable
        on either the prediction date or the trade date are skipped (not
        re-allocated to other stocks).

        :param current: The current portfolio position.
        :type current: Position
        :param trade_exchange: Exchange providing market data.
        :type trade_exchange: Exchange
        :param target_weight_position: ``{stock_id: weight}`` mapping.
        :type target_weight_position: dict
        :param risk_degree: Fraction of portfolio allocated to stocks.
        :type risk_degree: float
        :param pred_start_time: Start of the prediction window.
        :type pred_start_time: pd.Timestamp
        :param pred_end_time: End of the prediction window.
        :type pred_end_time: pd.Timestamp
        :param trade_start_time: Start of the trading window.
        :type trade_start_time: pd.Timestamp
        :param trade_end_time: End of the trading window.
        :type trade_end_time: pd.Timestamp

        :rtype: list
        :returns: A list of ``Order`` objects.
        """
        if target_weight_position is None:
            return []

        risk_total_value = risk_degree * current.calculate_value()

        current_stock = current.get_stock_list()
        amount_dict = {}
        for stock_id in target_weight_position:
            # Current rule will ignore the stock that not hold and cannot be traded at predict date
            if trade_exchange.is_stock_tradable(
                stock_id=stock_id, start_time=trade_start_time, end_time=trade_end_time
            ) and trade_exchange.is_stock_tradable(
                stock_id=stock_id, start_time=pred_start_time, end_time=pred_end_time
            ):
                amount_dict[stock_id] = (
                    risk_total_value
                    * target_weight_position[stock_id]
                    / trade_exchange.get_close(stock_id, start_time=pred_start_time, end_time=pred_end_time)
                )
                # TODO: Qlib use None to represent trading suspension.
                #  So last close price can't be the estimated trading price.
                # Maybe a close price with forward fill will be a better solution.
            elif stock_id in current_stock:
                amount_dict[stock_id] = (
                    risk_total_value * target_weight_position[stock_id] / current.get_stock_price(stock_id)
                )
            else:
                continue
        order_list = trade_exchange.generate_order_for_target_amount_position(
            target_position=amount_dict,
            current_position=current.get_stock_amount_dict(),
            start_time=trade_start_time,
            end_time=trade_end_time,
        )
        return order_list
