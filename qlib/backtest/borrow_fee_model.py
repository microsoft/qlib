# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Borrow fee models for short selling in Qlib backtests."""

# pylint: disable=R1716,R0913,W0613,W0201,W0718

from abc import ABC, abstractmethod
from typing import Dict, Optional
import pandas as pd


class BaseBorrowFeeModel(ABC):
    """
    Abstract base class for modeling borrowing fees in short selling.
    """

    @abstractmethod
    def get_borrow_rate(self, stock_id: str, date: pd.Timestamp) -> float:
        """
        Get the borrowing rate for a specific stock on a specific date.

        Parameters
        ----------
        stock_id : str
            The stock identifier
        date : pd.Timestamp
            The date for which to get the rate

        Returns
        -------
        float
            Annual borrowing rate as decimal (e.g., 0.03 for 3%)
        """
        raise NotImplementedError

    @abstractmethod
    def calculate_daily_cost(self, positions: Dict, date: pd.Timestamp) -> float:
        """
        Calculate total daily borrowing cost for all short positions.

        Parameters
        ----------
        positions : Dict
            Dictionary of positions with amounts and prices
        date : pd.Timestamp
            The date for calculation

        Returns
        -------
        float
            Total daily borrowing cost
        """
        raise NotImplementedError


class FixedRateBorrowFeeModel(BaseBorrowFeeModel):
    """
    Simple borrowing fee model with fixed rates.
    """

    def __init__(
        self,
        default_rate: float = 0.03,
        stock_rates: Optional[Dict[str, float]] = None,
        hard_to_borrow_rate: float = 0.10,
        days_per_year: int = 365,
    ):
        """
        Initialize fixed rate borrow fee model.

        Parameters
        ----------
        default_rate : float
            Default annual borrowing rate for most stocks (default 3%)
        stock_rates : Dict[str, float], optional
            Specific rates for individual stocks
        hard_to_borrow_rate : float
            Rate for hard-to-borrow stocks (default 10%)
        """
        self.default_rate = default_rate
        self.stock_rates = stock_rates or {}
        self.hard_to_borrow_rate = hard_to_borrow_rate
        # Configurable: set days-per-year by region (252 for stocks, 365 for crypto)
        self.daily_divisor = int(days_per_year) if days_per_year and days_per_year > 0 else 365

    def set_days_per_year(self, n: int) -> None:
        """Set days-per-year divisor used to convert annual rate to daily."""
        try:  # pylint: disable=W0718  # robustness preferred; benign conversion
            n = int(n)
            if n > 0:
                self.daily_divisor = n
        except Exception:  # pylint: disable=W0718
            pass

    def get_borrow_rate(self, stock_id: str, date: pd.Timestamp) -> float:
        """Get annual borrowing rate for a stock."""
        if stock_id in self.stock_rates:
            return self.stock_rates[stock_id]
        return self.default_rate

    def calculate_daily_cost(self, positions: Dict, date: pd.Timestamp) -> float:
        """Calculate total daily borrowing cost."""
        total_cost = 0.0

        for stock_id, position_info in positions.items():
            # Fix #4: strictly filter non-stock keys
            if not self._is_valid_stock_id(stock_id):
                continue

            if isinstance(position_info, dict):
                amount = position_info.get("amount", 0)
                price = position_info.get("price", 0)

                if (amount < 0) and (price > 0):  # charge only valid short positions
                    annual_rate = self.get_borrow_rate(stock_id, date)
                    daily_rate = annual_rate / self.daily_divisor
                    short_value = abs(amount * price)
                    total_cost += short_value * daily_rate

        return total_cost

    def _is_valid_stock_id(self, stock_id: str) -> bool:
        """Check whether it's a valid stock identifier."""
        # Filter out known non-stock keys
        non_stock_keys = {"cash", "cash_delay", "now_account_value", "borrow_cost_accumulated", "short_proceeds"}
        if stock_id in non_stock_keys:
            return False

        # Additional check: valid stock ids typically have a certain format/length
        if (not isinstance(stock_id, str)) or (len(stock_id) < 4):
            return False

        return True


class DynamicBorrowFeeModel(BaseBorrowFeeModel):
    """
    Dynamic borrowing fee model based on market conditions and availability.
    """

    def __init__(
        self,
        rate_data: Optional[pd.DataFrame] = None,
        default_rate: float = 0.03,
        volatility_adjustment: bool = True,
        liquidity_adjustment: bool = True,
        days_per_year: int = 365,
    ):
        """
        Initialize dynamic borrow fee model.

        Parameters
        ----------
        rate_data : pd.DataFrame, optional
            Historical borrowing rate data with MultiIndex (date, stock_id)
        default_rate : float
            Default rate when no data available
        volatility_adjustment : bool
            Adjust rates based on stock volatility
        liquidity_adjustment : bool
            Adjust rates based on stock liquidity
        """
        self.rate_data = rate_data
        self.default_rate = default_rate
        self.volatility_adjustment = volatility_adjustment
        self.liquidity_adjustment = liquidity_adjustment
        # Configurable: set days-per-year by region (252 for stocks, 365 for crypto)
        self.daily_divisor = int(days_per_year) if days_per_year and days_per_year > 0 else 365

    def set_days_per_year(self, n: int) -> None:
        """Set days-per-year divisor used to convert annual rate to daily."""
        try:  # pylint: disable=W0718
            n = int(n)
            if n > 0:
                self.daily_divisor = n
        except Exception:  # pylint: disable=W0718
            pass

        # Cache for calculated rates
        self._rate_cache = {}

    def get_borrow_rate(self, stock_id: str, date: pd.Timestamp) -> float:
        """
        Get borrowing rate with dynamic adjustments.
        """
        cache_key = (stock_id, date)
        if cache_key in self._rate_cache:
            return self._rate_cache[cache_key]

        base_rate = self._get_base_rate(stock_id, date)

        # Apply adjustments
        if self.volatility_adjustment:
            base_rate *= self._get_volatility_multiplier(stock_id, date)

        if self.liquidity_adjustment:
            base_rate *= self._get_liquidity_multiplier(stock_id, date)

        # Cap the rate at reasonable levels
        final_rate = min(base_rate, 0.50)  # Cap at 50% annual
        self._rate_cache[cache_key] = final_rate

        return final_rate

    def _get_base_rate(self, stock_id: str, date: pd.Timestamp) -> float:
        """Get base borrowing rate from data if available, otherwise default."""
        if self.rate_data is not None:
            try:
                return self.rate_data.loc[(date, stock_id), "borrow_rate"]
            except (KeyError, IndexError):
                pass
        return self.default_rate

    def _get_volatility_multiplier(self, stock_id: str, date: pd.Timestamp) -> float:
        """Return volatility multiplier (placeholder=1.0)."""
        return 1.0

    def _get_liquidity_multiplier(self, stock_id: str, date: pd.Timestamp) -> float:
        """Return liquidity multiplier (placeholder=1.0)."""
        return 1.0

    def calculate_daily_cost(self, positions: Dict, date: pd.Timestamp) -> float:
        """Calculate total daily borrowing cost with dynamic rates."""
        total_cost = 0.0

        for stock_id, position_info in positions.items():
            # Fix #4: use unified stock id validation
            if not self._is_valid_stock_id(stock_id):
                continue

            if isinstance(position_info, dict):
                amount = position_info.get("amount", 0)
                price = position_info.get("price", 0)

                if (amount < 0) and (price > 0):  # Short position
                    annual_rate = self.get_borrow_rate(stock_id, date)
                    daily_rate = annual_rate / self.daily_divisor
                    short_value = abs(amount * price)
                    total_cost += short_value * daily_rate

        return total_cost

    def _is_valid_stock_id(self, stock_id: str) -> bool:
        """Check whether it's a valid stock identifier."""
        # Filter out known non-stock keys
        non_stock_keys = {"cash", "cash_delay", "now_account_value", "borrow_cost_accumulated", "short_proceeds"}
        if stock_id in non_stock_keys:
            return False

        # Additional check: valid stock ids typically have a certain format/length
        if (not isinstance(stock_id, str)) or (len(stock_id) < 4):
            return False

        return True


class TieredBorrowFeeModel(BaseBorrowFeeModel):
    """
    Tiered borrowing fee model based on position size and stock category.
    """

    def __init__(
        self,
        easy_to_borrow: set = None,
        hard_to_borrow: set = None,
        size_tiers: Optional[Dict[float, float]] = None,
        days_per_year: int = 365,
    ):
        """
        Initialize tiered borrow fee model.

        Parameters
        ----------
        easy_to_borrow : set
            Set of stock IDs that are easy to borrow
        hard_to_borrow : set
            Set of stock IDs that are hard to borrow
        size_tiers : Dict[float, float]
            Position size tiers and corresponding rate adjustments
            E.g., {100000: 1.0, 1000000: 1.2, 10000000: 1.5}
        """
        self.easy_to_borrow = easy_to_borrow or set()
        self.hard_to_borrow = hard_to_borrow or set()

        # Default tier structure
        self.size_tiers = size_tiers or {
            100000: 1.0,  # <$100k: base rate
            1000000: 1.2,  # $100k-$1M: 1.2x rate
            10000000: 1.5,  # $1M-$10M: 1.5x rate
            float("inf"): 2.0,  # >$10M: 2x rate
        }

        # Base rates by category
        self.easy_rate = 0.01  # 1% for easy-to-borrow
        self.normal_rate = 0.03  # 3% for normal
        self.hard_rate = 0.10  # 10% for hard-to-borrow

        # Configurable: set days-per-year by region (252 for stocks, 365 for crypto)
        self.daily_divisor = int(days_per_year) if days_per_year and days_per_year > 0 else 365

    def set_days_per_year(self, n: int) -> None:
        """Set days-per-year divisor used to convert annual rate to daily."""
        try:
            n = int(n)
            if n > 0:
                self.daily_divisor = n
        except Exception:
            pass

    def get_borrow_rate(self, stock_id: str, date: pd.Timestamp) -> float:
        """Get base borrowing rate by stock category."""
        if stock_id in self.easy_to_borrow:
            return self.easy_rate
        if stock_id in self.hard_to_borrow:
            return self.hard_rate
        return self.normal_rate

    def _get_size_multiplier(self, position_value: float) -> float:
        """Get rate multiplier based on position size."""
        for threshold, multiplier in sorted(self.size_tiers.items()):
            if position_value <= threshold:
                return multiplier
        return 2.0  # Default max multiplier

    def calculate_daily_cost(self, positions: Dict, date: pd.Timestamp) -> float:
        """Calculate daily cost with tiered rates."""
        total_cost = 0.0

        for stock_id, position_info in positions.items():
            # Fix #4: use unified stock id validation
            if not self._is_valid_stock_id(stock_id):
                continue

            if isinstance(position_info, dict):
                amount = position_info.get("amount", 0)
                price = position_info.get("price", 0)

                if (amount < 0) and (price > 0):  # Short position
                    annual_rate = self.get_borrow_rate(stock_id, date)
                    daily_rate = annual_rate / self.daily_divisor
                    short_value = abs(amount * price)
                    total_cost += short_value * daily_rate

        return total_cost

    def _is_valid_stock_id(self, stock_id: str) -> bool:
        """Check whether it's a valid stock identifier."""
        # Filter out known non-stock keys
        non_stock_keys = {"cash", "cash_delay", "now_account_value", "borrow_cost_accumulated", "short_proceeds"}
        if stock_id in non_stock_keys:
            return False

        # Additional check: valid stock ids typically have a certain format/length
        if (not isinstance(stock_id, str)) or (len(stock_id) < 4):
            return False

        return True
