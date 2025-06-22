import calendar as cal_module
import re
import os
from datetime import datetime, timedelta
from functools import wraps
from typing import Union, List

import pandas as pd


def format_output(func):
    """
    Decorator to handle output format conversion for date-returning methods.
    Supports return_string, return_dt parameters to control output format.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        return_string = kwargs.pop('return_string', False)
        return_dt = kwargs.pop('return_dt', False)
        result = func(*args, **kwargs)

        # Convert format based on parameters
        if isinstance(result, (list, pd.DatetimeIndex)):
            if return_string:
                return [date.strftime('%Y-%m-%d') if hasattr(date, 'strftime')
                        else pd.to_datetime(str(date)).strftime('%Y-%m-%d') for date in result]
            elif return_dt:
                return [pd.to_datetime(date) for date in result]
            else:
                # Default: return integer format (YYYYMMDD)
                return [int(date.strftime('%Y%m%d')) if hasattr(date, 'strftime')
                        else int(pd.to_datetime(str(date)).strftime('%Y%m%d')) for date in result]
        else:
            # Single date
            if return_string:
                return result.strftime('%Y-%m-%d') if hasattr(result, 'strftime') else str(result)
            elif return_dt:
                return pd.to_datetime(result) if not isinstance(result, datetime) else result
            else:
                # Default: return integer format (YYYYMMDD)
                return int(result.strftime('%Y%m%d')) if hasattr(result, 'strftime') else int(str(result))

    return wrapper


class Calendar:
    """
    Extended with comprehensive trading calendar functionality:
    1. Date list queries (holidays, business days)
    2. Date validation functions (is_biz_day, is_holiday, is_weekend)
    3. Date adjustment and calculation (adjust_date, advance_date)
    4. Schedule generation
    5. Output format control (integer/string/datetime)
    """

    # Class variable for non-weekend holidays (loaded once)
    non_weekend_holidays = None

    @classmethod
    def _load_non_weekend_holidays(cls):
        """Load non-weekend holidays from CSV file if not already loaded."""
        if cls.non_weekend_holidays is None:
            csv_path = os.path.join(os.path.dirname(__file__), 'non_weekend_holidays.csv')
            try:
                with open(csv_path, 'r') as f:
                    cls.non_weekend_holidays = set(int(line.strip()) for line in f if line.strip())
            except FileNotFoundError:
                # Fallback to empty set if file not found
                cls.non_weekend_holidays = set()

    def __init__(self, start_date: str, end_date: str):
        # Load non-weekend holidays if not already loaded
        self._load_non_weekend_holidays()
        
        # 使用 'B' 表示营业日频率，这是交易日的常用替代。
        self._dates = pd.date_range(start=start_date, end=end_date, freq='B')

        # Extended functionality: parse dates and store for additional methods
        self.start_date = self._parse_date(start_date)
        self.end_date = self._parse_date(end_date)

        # Generate all dates in range for efficient operations
        self._all_dates = pd.date_range(start=self.start_date, end=self.end_date, freq='D')

    def get_trading_days(self, start_date: str = None, end_date: str = None) -> pd.DatetimeIndex:
        """
        返回指定范围内的交易日。
        如果未提供 start_date 或 end_date，则使用日历的默认范围。
        
        Note: This method now uses the same business day logic as other methods,
        excluding both weekends and non-weekend holidays, but maintains 
        pd.DatetimeIndex return type for backward compatibility.
        """
        # Use the same date range logic as biz_days method
        if start_date is None and end_date is None:
            actual_start = self.start_date
            actual_end = self.end_date
        else:
            actual_start = start_date if start_date else self.start_date
            actual_end = end_date if end_date else self.end_date
        
        # Directly use biz_days method and convert to pd.DatetimeIndex
        business_days = self.biz_days(actual_start, actual_end, return_dt=True)
        return pd.DatetimeIndex(business_days)

    # Extended functionality methods
    def _parse_date(self, date_input: Union[str, int, datetime]) -> datetime:
        """Parse various date input formats to datetime object."""
        if isinstance(date_input, datetime):
            return date_input
        elif isinstance(date_input, int):
            # Format: YYYYMMDD
            date_str = str(date_input)
            return datetime.strptime(date_str, '%Y%m%d')
        elif isinstance(date_input, str):
            # Try different string formats
            for fmt in ['%Y-%m-%d', '%Y%m%d', '%Y/%m/%d']:
                try:
                    return datetime.strptime(date_input, fmt)
                except ValueError:
                    continue
            raise ValueError(f"Unable to parse date: {date_input}")
        else:
            raise ValueError(f"Unsupported date type: {type(date_input)}")

    def _date_to_int(self, date: datetime) -> int:
        """Convert datetime to YYYYMMDD integer format."""
        return int(date.strftime('%Y%m%d'))

    def is_weekend(self, date: Union[str, int, datetime]) -> bool:
        """Check if date is weekend (Saturday or Sunday)."""
        parsed_date = self._parse_date(date)
        return parsed_date.weekday() >= 5  # 5=Saturday, 6=Sunday

    def is_holiday(self, date: Union[str, int, datetime]) -> bool:
        """Check if date is a holiday (weekend or non-weekend holiday)."""
        return self.is_weekend(date) or self.is_non_weekend_holiday(date)
    
    def is_non_weekend_holiday(self, date: Union[str, int, datetime]) -> bool:
        """Check if date is a non-weekend holiday."""
        date_int = self._date_to_int(self._parse_date(date))
        return date_int in self.non_weekend_holidays

    def is_biz_day(self, date: Union[str, int, datetime]) -> bool:
        """Check if date is a business day (not a holiday)."""
        return not self.is_holiday(date)

    @format_output
    def holidays(self, start_date: Union[str, int, datetime],
                 end_date: Union[str, int, datetime],
                 include_weekends: bool = True, **kwargs) -> List:
        """
        Get list of holidays in specified date range.
        
        Args:
            start_date: Start date
            end_date: End date
            include_weekends: Whether to include weekends as holidays
            **kwargs: Format control (return_string, return_dt)
        """
        start = self._parse_date(start_date)
        end = self._parse_date(end_date)

        holidays_list = []
        current_date = start

        while current_date <= end:
            if self.is_holiday(current_date) or (include_weekends and self.is_weekend(current_date)):
                holidays_list.append(current_date)
            current_date += timedelta(days=1)

        return holidays_list

    @format_output
    def biz_days(self, start_date: Union[str, int, datetime],
                 end_date: Union[str, int, datetime], **kwargs) -> List:
        """
        Get list of business days in specified date range.
        
        Args:
            start_date: Start date
            end_date: End date
            **kwargs: Format control (return_string, return_dt)
        """
        start = self._parse_date(start_date)
        end = self._parse_date(end_date)

        biz_days_list = []
        current_date = start

        while current_date <= end:
            if self.is_biz_day(current_date):
                biz_days_list.append(current_date)
            current_date += timedelta(days=1)

        return biz_days_list

    @format_output
    def adjust_date(self, date: Union[str, int, datetime], convention: int = 0, **kwargs):
        """
        Adjust date to business day using specified convention.
        
        Args:
            date: Input date
            convention: 0=Following (forward), 2=Preceding (backward)
            **kwargs: Format control (return_string, return_dt)
        """
        # Validate convention parameter first
        if convention not in [0, 2]:
            raise ValueError(f"Unsupported convention: {convention}")

        parsed_date = self._parse_date(date)

        if self.is_biz_day(parsed_date):
            return parsed_date

        if convention == 0:  # Following
            while not self.is_biz_day(parsed_date):
                parsed_date += timedelta(days=1)
        elif convention == 2:  # Preceding
            while not self.is_biz_day(parsed_date):
                parsed_date -= timedelta(days=1)

        return parsed_date

    def _parse_period(self, period: str) -> tuple:
        """
        Parse period string like '2b', '1w', '1m', '-1m' into (number, unit).
        
        Returns:
            tuple: (number, unit) where unit is 'b'(business), 'w'(week), 'm'(month), 'y'(year)
        """
        match = re.match(r'^(-?\d+)([bwmy])$', period.lower())
        if not match:
            raise ValueError(f"Invalid period format: {period}")

        number = int(match.group(1))
        unit = match.group(2)
        return number, unit

    @format_output
    def advance_date(self, date: Union[str, int, datetime], period: str, **kwargs):
        """
        Advance date by specified period.
        
        Args:
            date: Input date
            period: Period string like '2b' (2 business days), '1w' (1 week), '1m' (1 month)
            **kwargs: Format control (return_string, return_dt)
        """
        parsed_date = self._parse_date(date)
        number, unit = self._parse_period(period)

        if unit == 'b':  # Business days
            current_date = parsed_date
            days_moved = 0
            direction = 1 if number > 0 else -1
            target_days = abs(number)

            while days_moved < target_days:
                current_date += timedelta(days=direction)
                if self.is_biz_day(current_date):
                    days_moved += 1

            return current_date

        elif unit == 'w':  # Weeks
            return parsed_date + timedelta(weeks=number)

        elif unit == 'm':  # Months
            # Simple month addition (may need more sophisticated handling)
            new_month = parsed_date.month + number
            new_year = parsed_date.year + (new_month - 1) // 12
            new_month = ((new_month - 1) % 12) + 1

            try:
                return parsed_date.replace(year=new_year, month=new_month)
            except ValueError:
                # Handle day overflow (e.g., Jan 31 + 1 month)
                max_day = cal_module.monthrange(new_year, new_month)[1]
                return parsed_date.replace(year=new_year, month=new_month, day=min(parsed_date.day, max_day))

        elif unit == 'y':  # Years
            try:
                return parsed_date.replace(year=parsed_date.year + number)
            except ValueError:
                # Handle leap year edge case (Feb 29)
                return parsed_date.replace(year=parsed_date.year + number, month=2, day=28)

        else:
            raise ValueError(f"Unsupported period unit: {unit}")

    @format_output
    def schedule(self, start_date: Union[str, int, datetime],
                 end_date: Union[str, int, datetime],
                 tenor: str, date_generation_rule: int = 1, **kwargs) -> List:
        """
        Generate schedule of dates between start and end with specified tenor.
        
        Args:
            start_date: Start date
            end_date: End date
            tenor: Interval period (e.g., '1w', '3b', '1m')
            date_generation_rule: 1=Forward, 2=Backward
            **kwargs: Format control (return_string, return_dt)
        """
        start = self._parse_date(start_date)
        end = self._parse_date(end_date)

        schedule_dates = []

        if date_generation_rule == 1:  # Forward
            current_date = start
            while current_date <= end:
                schedule_dates.append(current_date)
                current_date = self._parse_date(self.advance_date(current_date, tenor, return_dt=True))

        elif date_generation_rule == 2:  # Backward
            current_date = end
            while current_date >= start:
                schedule_dates.insert(0, current_date)
                # For backward, we need to subtract the period
                number, unit = self._parse_period(tenor)
                backward_period = f"{-number}{unit}"
                current_date = self._parse_date(self.advance_date(current_date, backward_period, return_dt=True))

        else:
            raise ValueError(f"Unsupported date_generation_rule: {date_generation_rule}")

        return schedule_dates

    def __repr__(self):
        return f"Calendar(start={self._dates[0].date()}, end={self._dates[-1].date()}, days={len(self._dates)})"
