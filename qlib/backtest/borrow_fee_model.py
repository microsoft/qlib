# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Dict, Optional, Union
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod


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
    
    def __init__(self, 
                 default_rate: float = 0.03,
                 stock_rates: Optional[Dict[str, float]] = None,
                 hard_to_borrow_rate: float = 0.10,
                 days_per_year: int = 365):
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
        # 可配置：按地区设置（股票 252，Crypto 365）
        self.daily_divisor = int(days_per_year) if days_per_year and days_per_year > 0 else 365

    def set_days_per_year(self, n: int) -> None:
        try:
            n = int(n)
            if n > 0:
                self.daily_divisor = n
        except Exception:
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
            # 修复 #4: 严格过滤非股票键
            if not self._is_valid_stock_id(stock_id):
                continue
                
            if isinstance(position_info, dict):
                amount = position_info.get("amount", 0)
                price = position_info.get("price", 0)
                
                if amount < 0 and price > 0:  # 只对有效的空头仓位计费
                    annual_rate = self.get_borrow_rate(stock_id, date)
                    daily_rate = annual_rate / self.daily_divisor
                    short_value = abs(amount * price)
                    total_cost += short_value * daily_rate
                    
        return total_cost
    
    def _is_valid_stock_id(self, stock_id: str) -> bool:
        """检查是否为有效的股票代码"""
        # 过滤掉所有已知的非股票键
        non_stock_keys = {
            "cash", "cash_delay", "now_account_value", 
            "borrow_cost_accumulated", "short_proceeds"
        }
        if stock_id in non_stock_keys:
            return False
            
        # 进一步检查：有效股票代码通常有固定格式
        if not isinstance(stock_id, str) or len(stock_id) < 4:
            return False
            
        return True


class DynamicBorrowFeeModel(BaseBorrowFeeModel):
    """
    Dynamic borrowing fee model based on market conditions and availability.
    """
    
    def __init__(self,
                 rate_data: Optional[pd.DataFrame] = None,
                 default_rate: float = 0.03,
                 volatility_adjustment: bool = True,
                 liquidity_adjustment: bool = True,
                 days_per_year: int = 365):
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
        # 可配置：按地区设置（股票 252，Crypto 365）
        self.daily_divisor = int(days_per_year) if days_per_year and days_per_year > 0 else 365
    
    def set_days_per_year(self, n: int) -> None:
        try:
            n = int(n)
            if n > 0:
                self.daily_divisor = n
        except Exception:
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
        """Get base rate from data or default."""
        if self.rate_data is not None:
            try:
                return self.rate_data.loc[(date, stock_id), "borrow_rate"]
            except (KeyError, IndexError):
                pass
        return self.default_rate
        
    def _get_volatility_multiplier(self, stock_id: str, date: pd.Timestamp) -> float:
        """
        Calculate volatility-based rate multiplier.
        Higher volatility -> Higher borrowing cost
        """
        # Placeholder - in practice, calculate from historical data
        return 1.0
        
    def _get_liquidity_multiplier(self, stock_id: str, date: pd.Timestamp) -> float:
        """
        Calculate liquidity-based rate multiplier.
        Lower liquidity -> Higher borrowing cost
        """
        # Placeholder - in practice, calculate from volume data
        return 1.0
        
    def calculate_daily_cost(self, positions: Dict, date: pd.Timestamp) -> float:
        """Calculate total daily borrowing cost with dynamic rates."""
        total_cost = 0.0
        
        for stock_id, position_info in positions.items():
            # 修复 #4: 使用统一的股票ID验证
            if not self._is_valid_stock_id(stock_id):
                continue
                
            if isinstance(position_info, dict):
                amount = position_info.get("amount", 0)
                price = position_info.get("price", 0)
                
                if amount < 0 and price > 0:  # Short position
                    annual_rate = self.get_borrow_rate(stock_id, date)
                    daily_rate = annual_rate / self.daily_divisor
                    short_value = abs(amount * price)
                    total_cost += short_value * daily_rate
                    
        return total_cost
    
    def _is_valid_stock_id(self, stock_id: str) -> bool:
        """检查是否为有效的股票代码"""
        # 过滤掉所有已知的非股票键
        non_stock_keys = {
            "cash", "cash_delay", "now_account_value", 
            "borrow_cost_accumulated", "short_proceeds"
        }
        if stock_id in non_stock_keys:
            return False
            
        # 进一步检查：有效股票代码通常有固定格式
        if not isinstance(stock_id, str) or len(stock_id) < 4:
            return False
            
        return True


class TieredBorrowFeeModel(BaseBorrowFeeModel):
    """
    Tiered borrowing fee model based on position size and stock category.
    """
    
    def __init__(self,
                 easy_to_borrow: set = None,
                 hard_to_borrow: set = None,
                 size_tiers: Optional[Dict[float, float]] = None,
                 days_per_year: int = 365):
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
            100000: 1.0,    # <$100k: base rate
            1000000: 1.2,   # $100k-$1M: 1.2x rate
            10000000: 1.5,  # $1M-$10M: 1.5x rate
            float('inf'): 2.0  # >$10M: 2x rate
        }
        
        # Base rates by category
        self.easy_rate = 0.01  # 1% for easy-to-borrow
        self.normal_rate = 0.03  # 3% for normal
        self.hard_rate = 0.10  # 10% for hard-to-borrow
        
        # 可配置：按地区设置（股票 252，Crypto 365）
        self.daily_divisor = int(days_per_year) if days_per_year and days_per_year > 0 else 365

    def set_days_per_year(self, n: int) -> None:
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
        elif stock_id in self.hard_to_borrow:
            return self.hard_rate
        else:
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
            # 修复 #4: 使用统一的股票ID验证
            if not self._is_valid_stock_id(stock_id):
                continue
                
            if isinstance(position_info, dict):
                amount = position_info.get("amount", 0)
                price = position_info.get("price", 0)
                
                if amount < 0 and price > 0:  # Short position
                    short_value = abs(amount * price)
                    
                    # Get base rate and apply size multiplier
                    base_rate = self.get_borrow_rate(stock_id, date)
                    size_mult = self._get_size_multiplier(short_value)
                    
                    annual_rate = base_rate * size_mult
                    daily_rate = annual_rate / self.daily_divisor
                    
                    total_cost += short_value * daily_rate
                    
        return total_cost
    
    def _is_valid_stock_id(self, stock_id: str) -> bool:
        """检查是否为有效的股票代码"""
        # 过滤掉所有已知的非股票键
        non_stock_keys = {
            "cash", "cash_delay", "now_account_value", 
            "borrow_cost_accumulated", "short_proceeds"
        }
        if stock_id in non_stock_keys:
            return False
            
        # 进一步检查：有效股票代码通常有固定格式
        if not isinstance(stock_id, str) or len(stock_id) < 4:
            return False
            
        return True