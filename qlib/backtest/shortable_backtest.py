# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Integration module for short-selling support in Qlib backtest.
This module provides the main executor and strategy components.
"""

from __future__ import annotations

from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import math
from qlib.backtest.executor import SimulatorExecutor
from qlib.backtest.utils import CommonInfrastructure
from qlib.backtest.decision import Order, OrderDir, TradeDecisionWO
from qlib.backtest.account import Account
from qlib.backtest.position import Position
from qlib.utils import init_instance_by_config

from .shortable_exchange import ShortableExchange
from .shortable_position import ShortablePosition
from .borrow_fee_model import FixedRateBorrowFeeModel, BaseBorrowFeeModel


class ShortableAccount(Account):
    """
    Account that supports short selling by handling cases where
    stocks don't exist in current position.
    """

    def _update_state_from_order(self, order, trade_val, cost, trade_price):
        """
        Override to handle short selling cases where stock may not exist in position.
        """
        # CRITICAL: Validate price
        if trade_price is None or not np.isfinite(trade_price) or trade_price <= 0:
            return

        if self.is_port_metr_enabled():
            self.accum_info.add_turnover(abs(trade_val))  # Use absolute value for turnover
            self.accum_info.add_cost(cost)

        trade_amount = trade_val / trade_price

        if order.direction == OrderDir.SELL:
            # For short selling, stock may not exist in position
            try:
                p0 = self.current_position.get_stock_price(order.stock_id)
                profit = (trade_val - p0 * trade_amount) if (p0 is not None and np.isfinite(p0) and p0 > 0) else 0.0
            except (KeyError, AttributeError):
                profit = 0.0

            if self.is_port_metr_enabled():
                self.accum_info.add_return_value(profit)  # note here do not consider cost

        elif order.direction == OrderDir.BUY:
            try:
                p0 = self.current_position.get_stock_price(order.stock_id)
                profit = (p0 * trade_amount - trade_val) if (p0 is not None and np.isfinite(p0) and p0 > 0) else 0.0
            except (KeyError, AttributeError):
                profit = 0.0

            if self.is_port_metr_enabled():
                self.accum_info.add_return_value(profit)  # note here do not consider cost

    def get_portfolio_metrics(self):
        """Extend parent metrics with long/short-specific fields while keeping return shape unchanged."""
        try:
            df, meta = super().get_portfolio_metrics()
        except Exception:
            pm = super().get_portfolio_metrics()
            if isinstance(pm, tuple) and len(pm) == 2:
                df, meta = pm
            else:
                df, meta = None, pm if isinstance(pm, dict) else {}

        try:
            pos = self.current_position
            if isinstance(pos, ShortablePosition):
                extra = {
                    "leverage": pos.get_leverage(),
                    "net_exposure": pos.get_net_exposure(),
                    "total_borrow_cost": pos.borrow_cost_accumulated,
                }
                meta = {**(meta or {}), **extra}
        except Exception:
            pass

        return df, meta


class ShortableExecutor(SimulatorExecutor):
    """
    Executor that supports short selling with proper position and fee management.
    """

    def __init__(
        self,
        time_per_step: str = "day",
        generate_portfolio_metrics: bool = False,
        verbose: bool = False,
        track_data: bool = False,
        trade_exchange: Optional[ShortableExchange] = None,
        borrow_fee_model: Optional[BaseBorrowFeeModel] = None,
        settle_type: str = Position.ST_NO,
        region: str = "cn",  # Tweak #3: parameterize region to follow Qlib standard
        account: Optional[ShortableAccount] = None,
        common_infra: Optional[CommonInfrastructure] = None,
        **kwargs,
    ):
        """
        Initialize ShortableExecutor.

        Parameters
        ----------
        time_per_step : str
            Trading frequency
        generate_portfolio_metrics : bool
            Whether to generate portfolio metrics
        verbose : bool
            Print detailed execution info
        track_data : bool
            Track detailed trading data
        trade_exchange : ShortableExchange
            Exchange instance supporting short selling
        borrow_fee_model : BaseBorrowFeeModel
            Model for calculating borrowing fees
        settle_type : str
            Settlement type for positions
        region : str
            Region for trading calendar ('cn', 'us', etc.) - follows qlib.init() default
        """
        # Set attributes before calling parent __init__ because parent will invoke reset()
        self.settle_type = settle_type
        self.borrow_fee_model = borrow_fee_model or FixedRateBorrowFeeModel()
        self.region = region

        # Initialize trade_exchange if it's a config dict
        if isinstance(trade_exchange, dict):
            trade_exchange = init_instance_by_config(trade_exchange)

        super().__init__(
            time_per_step=time_per_step,
            generate_portfolio_metrics=generate_portfolio_metrics,
            verbose=verbose,
            track_data=track_data,
            trade_exchange=trade_exchange,
            settle_type=settle_type,
            common_infra=common_infra,
            **kwargs,
        )

        # Configure days-per-year for borrow fee (252 for stocks, 365 for crypto)
        try:
            if hasattr(self.borrow_fee_model, "set_days_per_year"):
                self.borrow_fee_model.set_days_per_year(365 if self.region == "crypto" else 252)
        except Exception:
            pass

    def reset(self, start_time=None, end_time=None, init_cash=1e6, **kwargs):
        """
        Reset executor time window. Position adaptation is handled in reset_common_infra when account is ready.
        """
        super().reset(start_time=start_time, end_time=end_time, init_cash=init_cash, **kwargs)

    def reset_common_infra(self, common_infra: CommonInfrastructure, copy_trade_account: bool = False) -> None:
        """Ensure account exists first, then adapt position to ShortablePosition and monkey-patch account hooks."""
        super().reset_common_infra(common_infra, copy_trade_account=copy_trade_account)
        if not hasattr(self, "trade_account") or self.trade_account is None:
            return
        # Replace current position with ShortablePosition (preserve holdings and cash)
        old_pos = self.trade_account.current_position
        position_dict = {}
        try:
            if hasattr(old_pos, "get_stock_list"):
                for sid in old_pos.get_stock_list():
                    position_dict[sid] = {
                        "amount": old_pos.get_stock_amount(sid),
                        "price": old_pos.get_stock_price(sid),
                    }
        except Exception:
            position_dict = {}

        # Determine a safe initial cash if old_pos has no get_cash
        try:
            fallback_cash = old_pos.get_cash(include_settle=True) if hasattr(old_pos, "get_cash") else None
        except Exception:
            fallback_cash = None
        if fallback_cash is None:
            try:
                fallback_cash = (
                    self.trade_account.current_position.get_cash()  # type: ignore[attr-defined]
                    if hasattr(self.trade_account.current_position, "get_cash")
                    else 1e6
                )
            except Exception:
                fallback_cash = 1e6

        pos = ShortablePosition(cash=fallback_cash, position_dict=position_dict)
        pos._settle_type = getattr(self, "settle_type", Position.ST_NO)
        self.trade_account.current_position = pos

        # Monkey-patch: use our fixed _update_state_from_order on existing account
        import types  # pylint: disable=C0415

        self.trade_account._update_state_from_order = types.MethodType(
            ShortableAccount._update_state_from_order, self.trade_account
        )
        # NOTE: Do not monkey-patch get_portfolio_metrics to avoid super() binding issues.

        # Sync aliases
        self.account = self.trade_account
        self.position = self.trade_account.current_position

    def _execute_orders(self, trade_decision: TradeDecisionWO, date: pd.Timestamp):
        """
        Execute orders with short-selling support and fee settlement.
        """
        # CRITICAL FIX: Mark-to-market all positions before trading
        # This ensures PnL is recognized daily, not just on trade days
        self._mark_to_market(date)

        # Execute orders normally
        trade_info = super()._execute_orders(trade_decision, date)

        # Post-check: ensure cash is non-negative
        if hasattr(self.account.current_position, "get_cash"):  # pylint: disable=has-member
            if self.account.current_position.get_cash() < -1e-6:
                if self.verbose:
                    print(f"[{date}] Warning: negative cash; check margin logic or scale weights")

        # Charge borrow fee once per trading day
        if self._is_trading_day(date) and isinstance(self.account.current_position, ShortablePosition):
            # CRITICAL FIX: use current market value instead of entry price for borrow fee
            position = self.account.current_position
            stock_positions = {}

            for stock_id in position.get_stock_list():
                info = position.position.get(stock_id, {})
                amt = info.get("amount", 0.0)

                # Skip non-short and zero positions
                if amt >= 0:
                    continue

                # Use current price (aligned with matching) instead of entry
                # For borrow fee, direction is not important; use BUY as a placeholder
                px = self.trade_exchange.get_deal_price(
                    stock_id=stock_id,
                    start_time=date,
                    end_time=date,
                    direction=OrderDir.BUY,  # Use OrderDir for consistency
                )

                # Robust fallback for borrow fee price
                if px is None or not np.isfinite(px) or px <= 0:
                    # Try position's last MTM price
                    px = position.get_stock_price(stock_id)

                if px is None or not np.isfinite(px) or px <= 0:
                    # Still no valid price; skip this stock
                    if self.verbose:
                        print(f"[{date}] Warning: Cannot get price for {stock_id}, skipping borrow fee")
                    continue

                # Use current market price or fallback
                stock_positions[stock_id] = {
                    "amount": amt,
                    "price": float(px),  # CRITICAL: Use daily market price or fallback
                }

            borrow_cost = self.borrow_fee_model.calculate_daily_cost(
                stock_positions, date  # Now with current daily prices
            )
            # Scale by step length (minute freq uses minutes-per-day proportion)
            try:
                borrow_cost *= self._borrow_fee_step_multiplier()
            except Exception:
                pass

            if borrow_cost > 0:
                self.account.current_position.add_borrow_cost(borrow_cost)
                if self.verbose:
                    print(f"[{date}] Daily borrowing cost: ${borrow_cost:.2f}")

        return trade_info

    def _mark_to_market(self, date: pd.Timestamp):
        """
        Mark all positions to market using current prices.
        This ensures daily PnL recognition.

        CRITICAL: Use same price calibration as trading (close or open)
        """
        if not isinstance(self.account.current_position, ShortablePosition):  # pylint: disable=has-member
            return

        position = self.account.current_position

        # Update price for all positions
        for stock_id in position.get_stock_list():
            if stock_id in position.position and isinstance(position.position[stock_id], dict):
                # Get current market price (use same calibration as trading)
                # For consistency, use close price if that's what we're trading at
                px = self.trade_exchange.get_deal_price(
                    stock_id=stock_id,
                    start_time=date,
                    end_time=date,
                    direction=OrderDir.BUY,  # Use OrderDir for consistency
                )

                if px is None or not np.isfinite(px) or px <= 0:
                    # Fallback to last valid price
                    px = position.get_stock_price(stock_id)

                if px is not None and np.isfinite(px) and px > 0:
                    # Update the position price to current market price
                    position.position[stock_id]["price"] = float(px)

        # This ensures PnL is calculated with current prices
        if self.verbose:
            equity = position.calculate_value()
            leverage = position.get_leverage()
            net_exp = position.get_net_exposure()
            print(f"[{date}] Mark-to-market: Equity=${equity:,.0f}, Leverage={leverage:.2f}, NetExp={net_exp:.2%}")

    def _is_trading_day(self, date):
        """Check whether it is a trading day.

        CRITICAL FIX: Only crypto markets trade 24/7, not US markets!
        """
        if self.region == "crypto":
            return True  # Crypto trades every day

        # For all other markets (including US), use trading calendar
        try:
            from qlib.data import D

            cal = D.calendar(freq=self.time_per_step, future=False)
            return date in cal
        except Exception:
            # Fallback: weekdays only for traditional markets
            return date.weekday() < 5

    def _borrow_fee_step_multiplier(self) -> float:
        """Convert per-day borrow fee to current step multiplier."""
        t = (self.time_per_step or "").lower()
        if t in ("day", "1d"):
            return 1.0
        try:
            import re

            m = re.match(r"(\d+)\s*min", t)
            if not m:
                return 1.0
            step_min = int(m.group(1))
            minutes_per_day = 1440 if self.region == "crypto" else 390
            if step_min <= 0:
                return 1.0
            return float(step_min) / float(minutes_per_day)
        except Exception:
            return 1.0

    def get_portfolio_metrics(self) -> Dict:
        """
        Get enhanced portfolio metrics including short-specific metrics.
        """
        metrics = super().get_portfolio_metrics()

        if isinstance(self.account.current_position, ShortablePosition):
            position = self.account.current_position

            # Add short-specific metrics
            metrics.update(
                {
                    "leverage": position.get_leverage(),
                    "net_exposure": position.get_net_exposure(),
                    "total_borrow_cost": position.borrow_cost_accumulated,  # read from attribute, not dict
                }
            )

            # Calculate long/short breakdown
            position_info = position.get_position_info()
            if not position_info.empty:
                long_positions = position_info[position_info["position_type"] == "long"]
                short_positions = position_info[position_info["position_type"] == "short"]

                metrics.update(
                    {
                        "long_value": long_positions["value"].sum() if not long_positions.empty else 0,
                        "short_value": short_positions["value"].abs().sum() if not short_positions.empty else 0,
                        "num_long_positions": len(long_positions),
                        "num_short_positions": len(short_positions),
                    }
                )

        return metrics


def round_to_lot(shares, lot=100):
    """Round towards zero by lot size to avoid exceeding limits."""
    if lot <= 1:
        return int(shares)  # toward zero
    lots = int(abs(shares) // lot)  # toward zero in lot units
    return int(math.copysign(lots * lot, shares))


class LongShortStrategy:
    """
    Long-short strategy that generates balanced long and short positions.
    """

    def __init__(
        self,
        gross_leverage: float = 1.6,
        net_exposure: float = 0.0,
        top_k: int = 30,
        exchange: Optional = None,
        risk_limit: Optional[Dict] = None,
        lot_size: Optional[int] = 100,
        min_trade_threshold: Optional[int] = 100,
    ):
        """
        Initialize long-short strategy.

        Parameters
        ----------
        gross_leverage : float
            Total leverage (long + short), e.g., 1.6 means 160% gross exposure
        net_exposure : float
            Net market exposure (long - short), e.g., 0.0 for market neutral
        top_k : int
            Number of stocks in each leg (long and short)
        exchange : Exchange
            Exchange instance for price queries
        risk_limit : Dict
            Risk limits (max_leverage, max_position_size, etc.)
        lot_size : int
            Trading lot size (default 100 for A-shares)
        min_trade_threshold : int
            Minimum trade threshold in shares (default 100)
        """
        self.gross_leverage = gross_leverage
        self.net_exposure = net_exposure
        self.top_k = top_k
        self.exchange = exchange
        # Allow None and treat intuitively: None -> no lot limit / no min threshold
        self.lot_size = 1 if lot_size is None else lot_size
        self.min_trade_threshold = 0 if min_trade_threshold is None else min_trade_threshold
        self.risk_limit = risk_limit or {
            "max_leverage": 2.0,
            "max_position_size": 0.1,
            "max_net_exposure": 0.3,
        }

        # Compute long/short ratios: gross = long + short, net = long - short
        # So: long = (gross + net) / 2, short = (gross - net) / 2
        self.long_ratio = (gross_leverage + net_exposure) / 2
        self.short_ratio = (gross_leverage - net_exposure) / 2

    def generate_trade_decision(
        self, signal: pd.Series, current_position: ShortablePosition, date: pd.Timestamp
    ) -> TradeDecisionWO:
        """
        Generate trade decisions based on signal using correct weight-to-shares conversion.
        """
        # Get current equity
        equity = current_position.calculate_value()

        # Select stocks
        signal_sorted = signal.sort_values(ascending=False)
        long_stocks = signal_sorted.head(self.top_k).index.tolist()
        short_stocks = signal_sorted.tail(self.top_k).index.tolist()

        # Fix #3: get prices by direction (consistent with matching)
        long_prices = self._get_current_prices(long_stocks, date, self.exchange, OrderDir.BUY) if long_stocks else {}
        short_prices = (
            self._get_current_prices(short_stocks, date, self.exchange, OrderDir.SELL) if short_stocks else {}
        )
        prices = {**long_prices, **short_prices}

        # Compute per-stock weights
        long_weight_per_stock = self.long_ratio / len(long_stocks) if long_stocks else 0
        short_weight_per_stock = -self.short_ratio / len(short_stocks) if short_stocks else 0  # negative

        # Tweak #2: hard cap per-position weight at equity × cap
        max_position_weight = self.risk_limit.get("max_position_size", 0.1)  # default 10%
        long_weight_per_stock = min(long_weight_per_stock, max_position_weight)
        short_weight_per_stock = max(short_weight_per_stock, -max_position_weight)  # negative, so use max

        orders = []

        # Long orders
        for stock in long_stocks:
            if stock in prices:
                target_shares = round_to_lot((long_weight_per_stock * equity) / prices[stock], lot=self.lot_size)
                current_shares = current_position.get_stock_amount(stock)
                delta = target_shares - current_shares

                if abs(delta) >= self.min_trade_threshold:  # respect configured trade threshold
                    direction = OrderDir.BUY if delta > 0 else OrderDir.SELL
                    orders.append(
                        Order(
                            stock_id=stock, amount=abs(int(delta)), direction=direction, start_time=date, end_time=date
                        )
                    )

        # Short orders
        for stock in short_stocks:
            if stock in prices:
                target_shares = round_to_lot(
                    (short_weight_per_stock * equity) / prices[stock], lot=self.lot_size  # negative
                )
                current_shares = current_position.get_stock_amount(stock)
                delta = target_shares - current_shares

                if abs(delta) >= self.min_trade_threshold:
                    direction = OrderDir.BUY if delta > 0 else OrderDir.SELL
                    orders.append(
                        Order(
                            stock_id=stock, amount=abs(int(delta)), direction=direction, start_time=date, end_time=date
                        )
                    )

        # Close positions not in target set
        current_stocks = set(current_position.get_stock_list())
        target_stocks = set(long_stocks + short_stocks)

        for stock in current_stocks - target_stocks:
            amount = current_position.get_stock_amount(stock)
            if abs(amount) >= self.min_trade_threshold:  # respect configured trade threshold
                direction = OrderDir.SELL if amount > 0 else OrderDir.BUY
                orders.append(
                    Order(stock_id=stock, amount=abs(int(amount)), direction=direction, start_time=date, end_time=date)
                )

        # Fix #2: enable risk limit checks
        if orders and not self._check_risk_limits(orders, current_position):
            # If exceeding risk limits, scale orders
            orders = self._scale_orders_for_risk(orders, current_position)

        # Note: The 2nd arg of TradeDecisionWO should be the strategy per Qlib design
        return TradeDecisionWO(orders, self)

    def _get_current_prices(self, stock_list, date, exchange=None, direction=None):
        """Fetch prices consistent with matching, supporting order direction."""
        prices = {}

        if exchange is not None:
            # Use exchange API to ensure consistency with matching
            for stock in stock_list:
                try:
                    # Fix #3: use direction-aware price fetching
                    price = exchange.get_deal_price(
                        stock_id=stock,
                        start_time=date,
                        end_time=date,
                        direction=direction,  # BUY/SELL direction, aligned with execution
                    )
                    if price is not None and not math.isnan(price):
                        prices[stock] = float(price)
                    else:
                        # Skip this stock if price unavailable
                        continue
                except Exception:
                    # Price fetch failed; skip
                    continue
        else:
            # Fallback: use a fixed price (testing only)
            for stock in stock_list:
                prices[stock] = 100.0  # placeholder

        return prices

    def _check_risk_limits(self, orders: List[Order], position: ShortablePosition) -> bool:
        """Check if orders comply with risk limits."""
        # Simulate position after orders
        simulated_position = self._simulate_position_change(orders, position)

        leverage = simulated_position.get_leverage()
        net_exposure = simulated_position.get_net_exposure()

        return leverage <= self.risk_limit["max_leverage"] and abs(net_exposure) <= self.risk_limit["max_net_exposure"]

    def _simulate_position_change(self, orders: List[Order], position: ShortablePosition) -> ShortablePosition:
        """Simulate position after executing orders with improved price sourcing."""
        stock_positions = {
            sid: {"amount": position.get_stock_amount(sid), "price": position.get_stock_price(sid)}
            for sid in position.get_stock_list()
        }

        sim = ShortablePosition(cash=position.get_cash(), position_dict=stock_positions)

        def _valid(p):
            return (p is not None) and np.isfinite(p) and (p > 0)

        for od in orders:
            cur = sim.get_stock_amount(od.stock_id)
            new_amt = cur + od.amount if od.direction == OrderDir.BUY else cur - od.amount

            # Try to get price: position price > exchange price; skip if can't get valid price
            price = sim.get_stock_price(od.stock_id) if od.stock_id in sim.position else None
            if not _valid(price) and getattr(self, "trade_exchange", None) is not None and hasattr(od, "start_time"):
                try:
                    px = self.trade_exchange.get_deal_price(
                        od.stock_id, od.start_time, od.end_time or od.start_time, od.direction
                    )
                    if _valid(px):
                        price = float(px)
                except Exception:
                    pass

            if not _valid(price):
                price = None  # Don't use placeholder 100, avoid misjudging leverage

            if od.stock_id not in sim.position:
                sim._init_stock(od.stock_id, new_amt, price if price is not None else 0.0)
            else:
                sim.position[od.stock_id]["amount"] = new_amt
                if price is not None:
                    sim.position[od.stock_id]["price"] = price

            # Only adjust cash with valid price (prevent placeholder from polluting risk control)
            if price is not None:
                if od.direction == OrderDir.BUY:
                    sim.position["cash"] -= price * od.amount
                else:
                    sim.position["cash"] += price * od.amount

        return sim

    def _scale_orders_for_risk(self, orders: List[Order], position: ShortablePosition) -> List[Order]:
        """Adaptive risk scaling - scale precisely by the degree of limit breach."""
        # Fix #2: simulate execution first to get leverage and net_exposure
        simulated_position = self._simulate_position_change(orders, position)
        leverage = simulated_position.get_leverage()
        net_exposure = abs(simulated_position.get_net_exposure())

        # Compute scale factor based on degree of breach
        max_leverage = self.risk_limit.get("max_leverage", 2.0)
        max_net_exposure = self.risk_limit.get("max_net_exposure", 0.3)

        scale_leverage = max_leverage / leverage if leverage > max_leverage else 1.0
        scale_net = max_net_exposure / net_exposure if net_exposure > max_net_exposure else 1.0

        # Take stricter constraint with a small safety margin
        scale_factor = min(scale_leverage, scale_net) * 0.98
        scale_factor = min(scale_factor, 1.0)  # never amplify, only shrink

        if scale_factor >= 0.99:  # scaling nearly unnecessary
            return orders

        scaled_orders = []
        for order in orders:
            # Round by lot size; keep original time fields
            scaled_amount = round_to_lot(order.amount * scale_factor, lot=self.lot_size)
            if scaled_amount <= 0:  # skip zero-after-rounding
                continue

            scaled_order = Order(
                stock_id=order.stock_id,
                amount=int(scaled_amount),
                direction=order.direction,
                start_time=order.start_time,
                end_time=order.end_time,
            )
            scaled_orders.append(scaled_order)

        return scaled_orders
