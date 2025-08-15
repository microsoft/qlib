# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Optional, Tuple, cast, TYPE_CHECKING
import numpy as np
import pandas as pd
from qlib.backtest.exchange import Exchange
from qlib.backtest.decision import Order
from qlib.backtest.position import BasePosition

if TYPE_CHECKING:
    from qlib.backtest.account import Account


class ShortableExchange(Exchange):
    """
    Exchange that supports short selling by removing the constraint
    that prevents selling more than current holdings.
    
    Key modifications:
    - Allows selling stocks not in current position (short selling)
    - Properly determines open/close costs based on position direction
    - Splits orders that cross zero position for accurate cost calculation
    - Maintains all other constraints (cash, volume limits, etc.)
    """
    
    def _calc_trade_info_by_order(
        self,
        order: Order,
        position: Optional[BasePosition],
        dealt_order_amount: dict,
    ) -> Tuple[float, float, float]:
        """
        Calculation of trade info with short selling support.
        
        **IMPORTANT**: Returns (trade_price, trade_val, trade_cost) to match parent class
        
        For BUY orders:
        - If current position < 0: covering short position -> use close_cost
        - If current position >= 0: opening/adding long position -> use open_cost
        - If crossing zero: split into cover short (close_cost) + open long (open_cost)
        
        For SELL orders:
        - If current position > 0: closing long position -> use close_cost  
        - If current position <= 0: opening/adding short position -> use open_cost
        - If crossing zero: split into close long (close_cost) + open short (open_cost)
        
        :param order: Order to be processed
        :param position: Current position (Optional)
        :param dealt_order_amount: Dict tracking dealt amounts {stock_id: float}
        :return: Tuple of (trade_price, trade_val, trade_cost)
        """
        
        # Get deal price first - with NaN/None guard
        trade_price = self.get_deal_price(order.stock_id, order.start_time, order.end_time, direction=order.direction)
        if trade_price is None or np.isnan(trade_price) or trade_price <= 0:
            self.logger.debug(f"Invalid price for {order.stock_id}, skipping order")
            order.deal_amount = 0
            return 0.0, 0.0, 0.0
        trade_price = cast(float, trade_price)
        
        # Calculate total market volume for impact cost - with NaN/None guard
        volume = self.get_volume(order.stock_id, order.start_time, order.end_time)
        if volume is None or np.isnan(volume):
            total_trade_val = 0.0
        else:
            total_trade_val = cast(float, volume) * trade_price
        
        # Set order factor for rounding
        order.factor = self.get_factor(order.stock_id, order.start_time, order.end_time)
        order.deal_amount = order.amount  # Start with full amount
        
        # Apply volume limits (common for both BUY and SELL)
        self._clip_amount_by_volume(order, dealt_order_amount)
        
        # Get current position amount
        current_amount = 0.0
        if position is not None and position.check_stock(order.stock_id):
            current_amount = position.get_stock_amount(order.stock_id)
        
        # Handle BUY orders
        if order.direction == Order.BUY:
            # Check if we're crossing zero (covering short then opening long)
            if current_amount < 0 and order.deal_amount > abs(current_amount):
                # Split into two legs: cover short + open long
                cover_amount = abs(current_amount)
                open_amount = order.deal_amount - cover_amount
                
                # Apply cash constraints for both legs (before rounding)
                if position is not None:
                    cash = position.get_cash(include_settle=True) if hasattr(position, "get_cash") else 0.0
                    
                    # Calculate costs for both legs (pre-rounding)
                    cover_val = cover_amount * trade_price
                    open_val = open_amount * trade_price
                    
                    # Initial impact cost calculation
                    if not total_trade_val or np.isnan(total_trade_val):
                        cover_impact = self.impact_cost
                        open_impact = self.impact_cost
                    else:
                        cover_impact = self.impact_cost * (cover_val / total_trade_val) ** 2
                        open_impact = self.impact_cost * (open_val / total_trade_val) ** 2
                    
                    # Calculate costs WITHOUT min_cost for each leg
                    cover_cost_no_min = cover_val * (self.close_cost + cover_impact)
                    open_cost_no_min = open_val * (self.open_cost + open_impact)
                    
                    # Apply min_cost ONCE for the total
                    total_cost = max(cover_cost_no_min + open_cost_no_min, self.min_cost)
                    total_val = cover_val + open_val
                    
                    # Check cash constraints
                    if cash < total_cost:
                        # Can't afford even the costs
                        order.deal_amount = 0
                        self.logger.debug(f"Order clipped due to cost higher than cash: {order}")
                    elif cash < total_val + total_cost:
                        # Need to reduce the open leg
                        available_for_open = cash - cover_val - cover_cost_no_min
                        if available_for_open > 0:
                            # Calculate max open amount considering the cost
                            max_open = self._get_buy_amount_by_cash_limit(
                                trade_price, available_for_open, self.open_cost + open_impact
                            )
                            open_amount = min(max_open, open_amount)
                            order.deal_amount = cover_amount + open_amount
                        else:
                            # Can only cover, not open new
                            order.deal_amount = cover_amount
                    
                    # Round the final amount
                    order.deal_amount = self.round_amount_by_trade_unit(order.deal_amount, order.factor)
                    
                    # Re-check cash constraints after rounding
                    final_val = order.deal_amount * trade_price
                    if order.deal_amount <= abs(current_amount):
                        # Only covering
                        if not total_trade_val or np.isnan(total_trade_val):
                            final_impact = self.impact_cost
                        else:
                            final_impact = self.impact_cost * (final_val / total_trade_val) ** 2
                        final_cost = max(final_val * (self.close_cost + final_impact), self.min_cost)
                    else:
                        # Still crossing zero after rounding
                        cover_amount = abs(current_amount)
                        open_amount = order.deal_amount - cover_amount
                        cover_val = cover_amount * trade_price
                        open_val = open_amount * trade_price
                        
                        final_cost = self._calc_cross_zero_cost_values(
                            cover_val,
                            open_val,
                            total_trade_val,
                            self.close_cost,
                            self.open_cost,
                        )
                    
                    # Final cash check after rounding with trade unit protection
                    if cash < final_val + final_cost:
                        # Reduce by one trade unit until it fits
                        trade_unit_amount = self._get_safe_trade_unit_amount(order)
                        if trade_unit_amount > 0:
                            steps = 0
                            max_steps = 10000  # Prevent infinite loop
                            while order.deal_amount > 0 and cash < order.deal_amount * trade_price + final_cost and steps < max_steps:
                                order.deal_amount -= trade_unit_amount
                                steps += 1
                                final_val = order.deal_amount * trade_price
                                # Recalculate cost with new amount
                                if order.deal_amount <= abs(current_amount):
                                    if not total_trade_val or np.isnan(total_trade_val):
                                        final_impact = self.impact_cost
                                    else:
                                        final_impact = self.impact_cost * (final_val / total_trade_val) ** 2
                                    final_cost = max(final_val * (self.close_cost + final_impact), self.min_cost)
                                else:
                                    cover_val = abs(current_amount) * trade_price
                                    open_val = (order.deal_amount - abs(current_amount)) * trade_price
                                    if not total_trade_val or np.isnan(total_trade_val):
                                        cover_impact = self.impact_cost
                                        open_impact = self.impact_cost
                                    else:
                                        cover_impact = self.impact_cost * (cover_val / total_trade_val) ** 2
                                        open_impact = self.impact_cost * (open_val / total_trade_val) ** 2
                                    cover_cost_no_min = cover_val * (self.close_cost + cover_impact)
                                    open_cost_no_min = open_val * (self.open_cost + open_impact)
                                    final_cost = max(cover_cost_no_min + open_cost_no_min, self.min_cost)
                            if steps >= max_steps:
                                self.logger.warning(f"Max iterations reached for order {order}, setting to 0")
                                order.deal_amount = 0
                        else:
                            order.deal_amount = 0
                else:
                    # No position info, just round
                    order.deal_amount = self.round_amount_by_trade_unit(order.deal_amount, order.factor)
                
                # Calculate final trade cost based on split legs
                trade_val = order.deal_amount * trade_price
                if order.deal_amount <= abs(current_amount):
                    # Only covering short
                    if not total_trade_val or np.isnan(total_trade_val):
                        adj_cost_ratio = self.impact_cost
                    else:
                        adj_cost_ratio = self.impact_cost * (trade_val / total_trade_val) ** 2
                    trade_cost = max(trade_val * (self.close_cost + adj_cost_ratio), self.min_cost) if trade_val > 1e-5 else 0
                else:
                    # Crossing zero: cover short + open long
                    cover_amount = abs(current_amount)
                    open_amount = order.deal_amount - cover_amount
                    cover_val = cover_amount * trade_price
                    open_val = open_amount * trade_price
                    
                    trade_cost = (
                        self._calc_cross_zero_cost_values(
                            cover_val,
                            open_val,
                            total_trade_val,
                            self.close_cost,
                            self.open_cost,
                        )
                        if trade_val > 1e-5
                        else 0
                    )
            
            else:
                # Simple case: either pure covering short or pure opening long
                if current_amount < 0:
                    # Covering short position - use close_cost
                    cost_ratio = self.close_cost
                else:
                    # Opening or adding to long position - use open_cost
                    cost_ratio = self.open_cost
                
                # Apply cash constraints
                if position is not None:
                    cash = position.get_cash(include_settle=True) if hasattr(position, "get_cash") else 0.0
                    trade_val = order.deal_amount * trade_price
                    
                    # Pre-calculate impact cost
                    if not total_trade_val or np.isnan(total_trade_val):
                        adj_cost_ratio = self.impact_cost
                    else:
                        adj_cost_ratio = self.impact_cost * (trade_val / total_trade_val) ** 2
                    
                    total_cost_ratio = cost_ratio + adj_cost_ratio
                    
                    if cash < max(trade_val * total_cost_ratio, self.min_cost):
                        # Cash cannot cover cost
                        order.deal_amount = 0
                        self.logger.debug(f"Order clipped due to cost higher than cash: {order}")
                    elif cash < trade_val + max(trade_val * total_cost_ratio, self.min_cost):
                        # Money is not enough for full order
                        max_buy_amount = self._get_buy_amount_by_cash_limit(trade_price, cash, total_cost_ratio)
                        order.deal_amount = min(max_buy_amount, order.deal_amount)
                        self.logger.debug(f"Order clipped due to cash limitation: {order}")
                    
                    # Round the amount
                    order.deal_amount = self.round_amount_by_trade_unit(order.deal_amount, order.factor)
                    
                    # Re-check cash constraint after rounding
                    final_val = order.deal_amount * trade_price
                    if not total_trade_val or np.isnan(total_trade_val):
                        final_impact = self.impact_cost
                    else:
                        final_impact = self.impact_cost * (final_val / total_trade_val) ** 2
                    final_cost = max(final_val * (cost_ratio + final_impact), self.min_cost)
                    
                    if cash < final_val + final_cost:
                        # Reduce by trade units until it fits
                        trade_unit_amount = self._get_safe_trade_unit_amount(order)
                        if trade_unit_amount > 0:
                            steps = 0
                            max_steps = 10000
                            while order.deal_amount > 0 and cash < order.deal_amount * trade_price + final_cost and steps < max_steps:
                                order.deal_amount -= trade_unit_amount
                                steps += 1
                                final_val = order.deal_amount * trade_price
                                if not total_trade_val or np.isnan(total_trade_val):
                                    final_impact = self.impact_cost
                                else:
                                    final_impact = self.impact_cost * (final_val / total_trade_val) ** 2
                                final_cost = max(final_val * (cost_ratio + final_impact), self.min_cost)
                            if steps >= max_steps:
                                self.logger.warning(f"Max iterations reached for order {order}, setting to 0")
                                order.deal_amount = 0
                        else:
                            order.deal_amount = 0
                else:
                    # Unknown amount of money - just round the amount
                    order.deal_amount = self.round_amount_by_trade_unit(order.deal_amount, order.factor)
                
                # Calculate final cost with final amount
                trade_val = order.deal_amount * trade_price
                if not total_trade_val or np.isnan(total_trade_val):
                    adj_cost_ratio = self.impact_cost
                else:
                    adj_cost_ratio = self.impact_cost * (trade_val / total_trade_val) ** 2
                trade_cost = max(trade_val * (cost_ratio + adj_cost_ratio), self.min_cost) if trade_val > 1e-5 else 0
        
        # Handle SELL orders
        elif order.direction == Order.SELL:
            # Check if we're crossing zero (closing long then opening short)
            if current_amount > 0 and order.deal_amount > current_amount:
                # Split into two legs: close long + open short
                close_amount = current_amount
                open_amount = order.deal_amount - current_amount
                
                # Apply cash constraint for transaction costs BEFORE rounding
                if position is not None:
                    cash = position.get_cash(include_settle=True) if hasattr(position, "get_cash") else 0.0
                    close_val = close_amount * trade_price
                    open_val = open_amount * trade_price
                    total_val = close_val + open_val
                    
                    # Calculate impact costs for both legs (pre-rounding)
                    total_cost = self._calc_cross_zero_cost_values(
                        close_val,
                        open_val,
                        total_trade_val,
                        self.close_cost,
                        self.open_cost,
                    )
                    
                    # Check if we have enough cash to pay transaction costs
                    # We receive cash from the sale but still need to pay costs
                    if cash + total_val < total_cost:
                        # Try to reduce the short leg
                        if cash + close_val >= max(close_cost_no_min, self.min_cost):
                            # Can at least close the long position
                            order.deal_amount = close_amount
                        else:
                            # Can't even close the position
                            order.deal_amount = 0
                            self.logger.debug(f"Order clipped due to insufficient cash for transaction costs: {order}")
                    else:
                        # Cash is sufficient, keep full amount
                        order.deal_amount = close_amount + open_amount
                    
                    # Now round both legs
                    if order.deal_amount > 0:
                        if order.deal_amount <= close_amount:
                            # Only closing, round the close amount
                            order.deal_amount = self.round_amount_by_trade_unit(order.deal_amount, order.factor)
                        else:
                            # Crossing zero, round both legs
                            close_amount = self.round_amount_by_trade_unit(close_amount, order.factor)
                            open_amount = self.round_amount_by_trade_unit(order.deal_amount - current_amount, order.factor)
                            order.deal_amount = close_amount + open_amount
                        
                        # Re-check cash constraint after rounding
                        final_val = order.deal_amount * trade_price
                        if order.deal_amount <= current_amount:
                            # Only closing
                            if not total_trade_val or np.isnan(total_trade_val):
                                final_impact = self.impact_cost
                            else:
                                final_impact = self.impact_cost * (final_val / total_trade_val) ** 2
                            final_cost = max(final_val * (self.close_cost + final_impact), self.min_cost)
                        else:
                            # Still crossing zero
                            close_val = current_amount * trade_price
                            open_val = (order.deal_amount - current_amount) * trade_price
                            final_cost = self._calc_cross_zero_cost_values(
                                close_val,
                                open_val,
                                total_trade_val,
                                self.close_cost,
                                self.open_cost,
                            )
                        
                        # Final check and potential reduction
                        if cash + final_val < final_cost:
                            # Try to reduce by trade units
                            trade_unit_amount = self._get_safe_trade_unit_amount(order)
                            if trade_unit_amount > 0:
                                steps = 0
                                max_steps = 10000
                                while order.deal_amount > 0 and cash + order.deal_amount * trade_price < final_cost and steps < max_steps:
                                    order.deal_amount -= trade_unit_amount
                                    steps += 1
                                    final_val = order.deal_amount * trade_price
                                    # Recalculate cost
                                    if order.deal_amount <= current_amount:
                                        if not total_trade_val or np.isnan(total_trade_val):
                                            final_impact = self.impact_cost
                                        else:
                                            final_impact = self.impact_cost * (final_val / total_trade_val) ** 2
                                        final_cost = max(final_val * (self.close_cost + final_impact), self.min_cost)
                                    else:
                                        close_val = current_amount * trade_price
                                        open_val = (order.deal_amount - current_amount) * trade_price
                                        if not total_trade_val or np.isnan(total_trade_val):
                                            close_impact = self.impact_cost
                                            open_impact = self.impact_cost
                                        else:
                                            close_impact = self.impact_cost * (close_val / total_trade_val) ** 2
                                            open_impact = self.impact_cost * (open_val / total_trade_val) ** 2
                                        close_cost_no_min = close_val * (self.close_cost + close_impact)
                                        open_cost_no_min = open_val * (self.open_cost + open_impact)
                                        final_cost = max(close_cost_no_min + open_cost_no_min, self.min_cost)
                                if steps >= max_steps:
                                    self.logger.warning(f"Max iterations reached for order {order}, setting to 0")
                                    order.deal_amount = 0
                            else:
                                order.deal_amount = 0
                else:
                    # No position info, just round
                    close_amount = self.round_amount_by_trade_unit(close_amount, order.factor)
                    open_amount = self.round_amount_by_trade_unit(open_amount, order.factor)
                    order.deal_amount = close_amount + open_amount
                
                # Calculate final trade cost based on split legs
                trade_val = order.deal_amount * trade_price
                if order.deal_amount <= current_amount:
                    # Only closing long
                    if not total_trade_val or np.isnan(total_trade_val):
                        adj_cost_ratio = self.impact_cost
                    else:
                        adj_cost_ratio = self.impact_cost * (trade_val / total_trade_val) ** 2
                    trade_cost = max(trade_val * (self.close_cost + adj_cost_ratio), self.min_cost) if trade_val > 1e-5 else 0
                else:
                    # Crossing zero: close long + open short
                    close_val = current_amount * trade_price
                    open_val = (order.deal_amount - current_amount) * trade_price
                    
                    trade_cost = (
                        self._calc_cross_zero_cost_values(
                            close_val,
                            open_val,
                            total_trade_val,
                            self.close_cost,
                            self.open_cost,
                        )
                        if trade_val > 1e-5
                        else 0
                    )
            
            else:
                # Simple case: either pure closing long or pure opening short
                if current_amount > 0:
                    # Closing long position - use close_cost
                    cost_ratio = self.close_cost
                    # Don't sell more than we have when closing long
                    order.deal_amount = min(current_amount, order.deal_amount)
                else:
                    # Opening or adding to short position - use open_cost
                    cost_ratio = self.open_cost
                    # No constraint on amount for short selling
                
                # Round the amount
                order.deal_amount = self.round_amount_by_trade_unit(order.deal_amount, order.factor)
                
                # Apply cash constraint for transaction costs
                if position is not None:
                    cash = position.get_cash(include_settle=True) if hasattr(position, "get_cash") else 0.0
                    trade_val = order.deal_amount * trade_price
                    
                    # Calculate impact cost with final amount
                    if not total_trade_val or np.isnan(total_trade_val):
                        adj_cost_ratio = self.impact_cost
                    else:
                        adj_cost_ratio = self.impact_cost * (trade_val / total_trade_val) ** 2
                    
                    expected_cost = max(trade_val * (cost_ratio + adj_cost_ratio), self.min_cost)
                    
                    # Check if we have enough cash to pay transaction costs
                    # For SELL orders, we receive cash from the sale but still need to pay costs
                    if cash + trade_val < expected_cost:
                        # Not enough cash to cover transaction costs even after receiving sale proceeds
                        order.deal_amount = 0
                        self.logger.debug(f"Order clipped due to insufficient cash for transaction costs: {order}")
                
                # Calculate final cost
                trade_val = order.deal_amount * trade_price
                if not total_trade_val or np.isnan(total_trade_val):
                    adj_cost_ratio = self.impact_cost
                else:
                    adj_cost_ratio = self.impact_cost * (trade_val / total_trade_val) ** 2
                trade_cost = max(trade_val * (cost_ratio + adj_cost_ratio), self.min_cost) if trade_val > 1e-5 else 0
        
        else:
            raise NotImplementedError(f"Order direction {order.direction} not supported")
        
        # Final trade value calculation
        trade_val = order.deal_amount * trade_price
        
        # CRITICAL: Return in correct order (trade_price, trade_val, trade_cost)
        return trade_price, trade_val, trade_cost

    def _get_safe_trade_unit_amount(self, order: Order) -> float:
        """获取安全的交易单位数量，避免无限循环或无意义的极小步长。
        返回 <=0 表示不可用。
        """
        try:
            tua = self.get_amount_of_trade_unit(order.factor, order.stock_id, order.start_time, order.end_time)
            if tua is None:
                return 0.0
            tua = float(tua)
            if not np.isfinite(tua) or tua <= 0 or tua < 1e-12:
                return 0.0
            return tua
        except Exception:
            return 0.0

    def _calc_cross_zero_cost_values(
        self,
        close_val: float,
        open_val: float,
        total_trade_val: Optional[float],
        close_cost_ratio: float,
        open_cost_ratio: float,
    ) -> float:
        """合并计算跨零两条腿的交易成本，并仅计一次 min_cost。"""
        if not total_trade_val or np.isnan(total_trade_val) or total_trade_val <= 0:
            close_impact = self.impact_cost
            open_impact = self.impact_cost
        else:
            close_impact = self.impact_cost * (close_val / total_trade_val) ** 2
            open_impact = self.impact_cost * (open_val / total_trade_val) ** 2
        close_cost_no_min = close_val * (close_cost_ratio + close_impact)
        open_cost_no_min = open_val * (open_cost_ratio + open_impact)
        return max(close_cost_no_min + open_cost_no_min, self.min_cost)
    
    def generate_amount_position_from_weight_position(
        self,
        weight_position: dict,
        cash: float,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        round_amount: bool = True,
        verbose: bool = False,
        account: "Account" = None,
        gross_leverage: float = 1.0,
    ) -> dict:
        """
        Generate amount position from weight position with support for negative weights (short positions).
        
        Uses absolute weight normalization to avoid "double spending" cash on long and short positions.
        
        :param weight_position: Dict of {stock_id: weight}, weights can be negative for short positions
        :param cash: Available cash
        :param start_time: Start time for the trading period
        :param end_time: End time for the trading period  
        :param round_amount: Whether to round amounts to trading units
        :param verbose: Whether to print debug information
        :param account: Account object (optional)
        :param gross_leverage: Gross leverage factor (default 1.0). 
                              Total position value = cash * gross_leverage
        :return: Dict of {stock_id: amount}, negative amounts indicate short positions
        """
        
        # Calculate total absolute weight for normalization
        total_abs_weight = sum(abs(w) for w in weight_position.values())
        
        if total_abs_weight == 0:
            return {}
        
        amount_position = {}
        
        # Process all positions using absolute weight normalization
        for stock_id, weight in weight_position.items():
            if self.is_stock_tradable(stock_id, start_time, end_time):
                # Determine order direction based on weight sign
                if weight > 0:
                    price = self.get_deal_price(stock_id, start_time, end_time, Order.BUY)
                else:
                    price = self.get_deal_price(stock_id, start_time, end_time, Order.SELL)
                
                # Price protection: skip if price is invalid
                if not price or np.isnan(price) or price <= 0:
                    self.logger.debug(f"Invalid price for {stock_id}, skipping position generation")
                    continue
                
                # Calculate target value using absolute weight normalization
                target_value = cash * (abs(weight) / total_abs_weight) * gross_leverage
                
                # Calculate target amount (positive for long, negative for short)
                if weight > 0:
                    target_amount = target_value / price
                else:
                    target_amount = -target_value / price
                
                if round_amount:
                    factor = self.get_factor(stock_id, start_time, end_time)
                    if target_amount > 0:
                        target_amount = self.round_amount_by_trade_unit(target_amount, factor)
                    else:
                        # Round the absolute value then make it negative again
                        target_amount = -self.round_amount_by_trade_unit(abs(target_amount), factor)
                
                amount_position[stock_id] = target_amount
        
        if verbose:
            self.logger.info(f"Generated amount position with gross leverage {gross_leverage}: {amount_position}")
        
        return amount_position