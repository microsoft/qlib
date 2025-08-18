# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Dict, Union, Optional
import numpy as np
import pandas as pd
from qlib.backtest.position import Position


class ShortablePosition(Position):
    """
    Position that supports negative holdings (short positions).

    Key differences from standard Position:
    1. Allows negative amounts for stocks (short positions)
    2. Properly calculates value for both long and short positions
    3. Tracks borrowing costs and other short-related metrics
    4. Maintains cash settlement consistency with qlib
    """

    # Class constant for position close tolerance
    POSITION_EPSILON = 1e-10  # Can be adjusted based on trade unit requirements

    def __init__(
        self,
        cash: float = 0,
        position_dict: Dict[str, Union[Dict[str, float], float]] = None,
        borrow_rate: float = 0.03,
    ):  # Annual borrowing rate, default 3%
        """
        Initialize ShortablePosition.

        Parameters
        ----------
        cash : float
            Initial cash
        position_dict : dict
            Initial positions (can include negative amounts for shorts)
        borrow_rate : float
            Annual rate for borrowing stocks (as decimal, e.g., 0.03 for 3%)
        """
        # Initialize our attributes BEFORE calling super().__init__
        # because super().__init__ will call calculate_value() which needs these
        self.borrow_rate = borrow_rate
        self._daily_borrow_rate = borrow_rate / 252  # Convert to daily rate
        self.borrow_cost_accumulated = 0.0
        self.short_proceeds = {}  # Track proceeds from short sales {stock_id: proceeds}

        # Initialize logger if available
        try:
            from qlib.log import get_module_logger

            self.logger = get_module_logger("ShortablePosition")
        except ImportError:
            self.logger = None

        # Handle default parameter
        if position_dict is None:
            position_dict = {}

        # Now call parent init which will use our calculate_value() method
        super().__init__(cash=cash, position_dict=position_dict)

        # Ensure cash_delay exists for robustness
        self.position.setdefault("cash_delay", 0.0)

    def _sell_stock(self, stock_id: str, trade_val: float, cost: float, trade_price: float) -> None:
        """
        Sell stock, allowing short positions.

        This overrides the parent method to allow negative positions.
        """
        trade_amount = trade_val / trade_price

        if stock_id not in self.position:
            # Opening a new short position
            self._init_stock(stock_id=stock_id, amount=-trade_amount, price=trade_price)
            # Track short sale proceeds
            self.short_proceeds[stock_id] = trade_val
        else:
            current_amount = self.position[stock_id]["amount"]
            new_amount = current_amount - trade_amount

            # Use absolute tolerance for position close check
            if abs(new_amount) < self.POSITION_EPSILON:
                # Position closed
                self._del_stock(stock_id)
                if stock_id in self.short_proceeds:
                    del self.short_proceeds[stock_id]
            else:
                # Update position (can go negative)
                self.position[stock_id]["amount"] = new_amount
                self.position[stock_id]["price"] = trade_price  # Update price on trade

                # Track short proceeds for new or increased short positions
                if new_amount < 0:
                    if current_amount >= 0:
                        # Going from long to short: record short portion proceeds
                        short_amount = abs(new_amount)
                        self.short_proceeds[stock_id] = short_amount * trade_price
                    else:
                        # Increasing short position: accumulate new short proceeds
                        if stock_id not in self.short_proceeds:
                            self.short_proceeds[stock_id] = 0
                        # Only accumulate the additional short portion
                        # More explicit calculation for robustness
                        additional_short_amount = max(0.0, -(new_amount - current_amount))
                        self.short_proceeds[stock_id] += additional_short_amount * trade_price

        # Update cash
        new_cash = trade_val - cost
        if self._settle_type == self.ST_CASH:
            self.position["cash_delay"] += new_cash
        elif self._settle_type == self.ST_NO:
            self.position["cash"] += new_cash
        else:
            raise NotImplementedError(f"This type of input is not supported")

    def _buy_stock(self, stock_id: str, trade_val: float, cost: float, trade_price: float) -> None:
        """
        Buy stock, which can also mean covering a short position.

        CRITICAL FIX: Buy orders immediately reduce cash (not delayed), consistent with qlib.
        """
        trade_amount = trade_val / trade_price

        if stock_id not in self.position:
            # Opening new long position
            self._init_stock(stock_id=stock_id, amount=trade_amount, price=trade_price)
        else:
            current_amount = self.position[stock_id]["amount"]

            if current_amount < 0:
                # Covering a short position
                new_amount = current_amount + trade_amount

                # CRITICAL FIX: Reduce short_proceeds when partially covering
                covered_amount = min(trade_amount, abs(current_amount))
                if stock_id in self.short_proceeds and covered_amount > 0:
                    if abs(current_amount) > 0:
                        reduction_ratio = covered_amount / abs(current_amount)
                        self.short_proceeds[stock_id] *= 1 - reduction_ratio
                        if self.short_proceeds[stock_id] < self.POSITION_EPSILON:
                            del self.short_proceeds[stock_id]

                if new_amount >= 0:
                    # Fully covered and possibly going long
                    if stock_id in self.short_proceeds:
                        del self.short_proceeds[stock_id]

                # Use absolute tolerance for position close check
                if abs(new_amount) < self.POSITION_EPSILON:
                    # Position fully closed
                    self._del_stock(stock_id)
                else:
                    self.position[stock_id]["amount"] = new_amount
                    self.position[stock_id]["price"] = trade_price  # Update price on trade
            else:
                # Adding to long position
                self.position[stock_id]["amount"] += trade_amount
                self.position[stock_id]["price"] = trade_price  # Update price on trade

        # CRITICAL FIX: Buy orders immediately reduce cash (not delayed)
        # This is consistent with qlib's implementation and prevents over-buying
        self.position["cash"] -= trade_val + cost

    def calculate_stock_value(self) -> float:
        """
        Calculate total value of stock positions.

        For long positions: value = amount * price
        For short positions: value = amount * price (negative)
        """
        stock_list = self.get_stock_list()
        value = 0

        for stock_id in stock_list:
            amount = self.position[stock_id]["amount"]
            price = self.position[stock_id].get("price", 0)
            # Price robustness check
            if price is not None and np.isfinite(price) and price > 0:
                value += amount * price  # Negative for shorts
            elif price is None or not np.isfinite(price) or price <= 0:
                # Log for debugging if logger is available
                if getattr(self, "logger", None) is not None:
                    self.logger.debug(f"Invalid price for {stock_id}: {price}")

        return value

    def get_cash(self, include_settle: bool = False) -> float:
        """
        Get available cash.

        CRITICAL FIX: Added include_settle parameter to match parent class interface.

        Parameters
        ----------
        include_settle : bool
            If True, include cash_delay (pending settlements) in the returned value

        Returns
        -------
        float
            Available cash (optionally including pending settlements)
        """
        cash = self.position.get("cash", 0.0)
        if include_settle:
            cash += self.position.get("cash_delay", 0.0)
        return cash

    def set_cash(self, value: float) -> None:
        """
        Set cash value directly.

        Parameters
        ----------
        value : float
            New cash value
        """
        self.position["cash"] = float(value)

    def add_borrow_cost(self, cost: float) -> None:
        """
        Deduct borrowing cost from cash and track accumulated costs.

        Parameters
        ----------
        cost : float
            Borrowing cost to deduct
        """
        self.position["cash"] -= float(cost)
        self.borrow_cost_accumulated += float(cost)

    def calculate_value(self) -> float:
        """
        Calculate total portfolio value.

        Total value = cash + cash_delay + stock_value
        Borrowing costs are already deducted from cash, so not subtracted again.
        """
        stock_value = self.calculate_stock_value()
        cash = self.position.get("cash", 0.0)
        cash_delay = self.position.get("cash_delay", 0.0)

        return cash + cash_delay + stock_value

    def get_leverage(self) -> float:
        """
        Calculate portfolio leverage.

        Leverage = (Long Value + |Short Value|) / Total Equity

        Returns
        -------
        float
            Portfolio leverage ratio
        """
        stock_list = self.get_stock_list()
        long_value = 0
        short_value = 0

        for stock_id in stock_list:
            if isinstance(self.position[stock_id], dict):
                amount = self.position[stock_id].get("amount", 0)
                price = self.position[stock_id].get("price", 0)
                # Price robustness check
                if price is not None and np.isfinite(price) and price > 0:
                    position_value = amount * price

                    if amount > 0:
                        long_value += position_value
                    else:
                        short_value += abs(position_value)

        total_equity = self.calculate_value()
        if total_equity <= 0:
            return np.inf

        gross_exposure = long_value + short_value
        return gross_exposure / total_equity

    def get_net_exposure(self) -> float:
        """
        Calculate net market exposure.

        Net Exposure = (Long Value - Short Value) / Total Equity

        Returns
        -------
        float
            Net exposure ratio
        """
        stock_list = self.get_stock_list()
        long_value = 0
        short_value = 0

        for stock_id in stock_list:
            if isinstance(self.position[stock_id], dict):
                amount = self.position[stock_id].get("amount", 0)
                price = self.position[stock_id].get("price", 0)
                # Price robustness check
                if price is not None and np.isfinite(price) and price > 0:
                    position_value = amount * price

                    if amount > 0:
                        long_value += position_value
                    else:
                        short_value += abs(position_value)

        total_equity = self.calculate_value()
        if total_equity <= 0:
            return 0

        net_exposure = (long_value - short_value) / total_equity
        return net_exposure

    def calculate_daily_borrow_cost(self) -> float:
        """
        Calculate daily borrowing cost for short positions.

        Returns
        -------
        float
            Daily borrowing cost
        """
        stock_list = self.get_stock_list()
        daily_cost = 0

        for stock_id in stock_list:
            if isinstance(self.position[stock_id], dict):
                amount = self.position[stock_id].get("amount", 0)
                if amount < 0:  # Short position
                    price = self.position[stock_id].get("price", 0)
                    # Price robustness check
                    if price is not None and np.isfinite(price) and price > 0:
                        short_value = abs(amount * price)
                        daily_cost += short_value * self._daily_borrow_rate
                    elif price is None or not np.isfinite(price) or price <= 0:
                        if getattr(self, "logger", None) is not None:
                            self.logger.debug(f"Invalid price for short position {stock_id}: {price}")

        return daily_cost

    def settle_daily_costs(self) -> None:
        """
        Settle daily costs including borrowing fees.
        Should be called at the end of each trading day.

        Note: Consider using add_borrow_cost() for more control.
        """
        borrow_cost = self.calculate_daily_borrow_cost()
        if borrow_cost > 0:
            self.add_borrow_cost(borrow_cost)

    def get_position_info(self) -> pd.DataFrame:
        """
        Get detailed position information as DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with position details including:
            - amount: position size (negative for shorts)
            - price: current price
            - value: position value
            - weight: position weight in portfolio
            - position_type: "long" or "short"
        """
        data = []
        stock_list = self.get_stock_list()

        for stock_id in stock_list:
            amount = self.position[stock_id]["amount"]
            price = self.position[stock_id].get("price", 0)
            weight = self.position[stock_id].get("weight", 0)

            # Price robustness check
            if price is not None and np.isfinite(price) and price > 0:
                value = amount * price
            else:
                value = 0  # Cannot calculate value without valid price

            data.append(
                {
                    "stock_id": stock_id,
                    "amount": amount,
                    "price": price if price is not None else 0,
                    "value": value,
                    "weight": weight,
                    "position_type": "long" if amount > 0 else "short",
                }
            )

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df = df.set_index("stock_id")
        return df

    def get_short_positions(self) -> Dict[str, float]:
        """
        Get all short positions.

        Returns
        -------
        dict
            Dictionary of {stock_id: amount} for all short positions
        """
        shorts = {}
        stock_list = self.get_stock_list()

        for stock_id in stock_list:
            amount = self.position[stock_id]["amount"]
            if amount < 0:
                shorts[stock_id] = amount

        return shorts

    def get_long_positions(self) -> Dict[str, float]:
        """
        Get all long positions.

        Returns
        -------
        dict
            Dictionary of {stock_id: amount} for all long positions
        """
        longs = {}
        stock_list = self.get_stock_list()

        for stock_id in stock_list:
            amount = self.position[stock_id]["amount"]
            if amount > 0:
                longs[stock_id] = amount

        return longs

    def get_gross_value(self) -> float:
        """
        Get gross portfolio value (sum of absolute values of all positions).

        Returns
        -------
        float
            Gross portfolio value
        """
        gross = 0.0
        for sid in self.get_stock_list():
            pos = self.position[sid]
            amt = pos.get("amount", 0.0)
            price = pos.get("price", None)
            if price is not None and np.isfinite(price) and price > 0:
                gross += abs(amt * price)
            elif price is None or not np.isfinite(price) or price <= 0:
                if getattr(self, "logger", None) is not None:
                    self.logger.debug(f"Invalid price for {sid} in gross value calculation: {price}")
        return gross

    def get_net_value(self) -> float:
        """
        Get net portfolio value (long value - short value).

        Returns
        -------
        float
            Net portfolio value
        """
        return self.calculate_stock_value()

    def update_all_stock_prices(self, price_dict: Dict[str, float]) -> None:
        """
        Update prices for all positions (mark-to-market).

        This should be called at the end of each trading day with closing prices
        to ensure accurate portfolio valuation.

        Parameters
        ----------
        price_dict : dict
            Dictionary of {stock_id: price} with current market prices
        """
        for stock_id in self.get_stock_list():
            if stock_id in price_dict:
                price = price_dict[stock_id]
                if price is not None and np.isfinite(price) and price > 0:
                    self.position[stock_id]["price"] = price

    def __str__(self) -> str:
        """String representation showing position details."""
        # Handle potential inf values safely
        leverage = self.get_leverage()
        leverage_str = round(leverage, 2) if np.isfinite(leverage) else "inf"

        net_exp = self.get_net_exposure()
        net_exp_str = round(net_exp, 2) if np.isfinite(net_exp) else "inf"

        info = {
            "cash": self.get_cash(),
            "cash_delay": self.position.get("cash_delay", 0),
            "stock_value": self.calculate_stock_value(),
            "total_value": self.calculate_value(),
            "leverage": leverage_str,
            "net_exposure": net_exp_str,
            "long_positions": len(self.get_long_positions()),
            "short_positions": len(self.get_short_positions()),
            "borrow_cost_accumulated": round(self.borrow_cost_accumulated, 2),
        }
        return f"ShortablePosition({info})"
