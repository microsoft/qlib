import pytest
import pandas as pd
from datetime import datetime
from factor_engine.data_layer.calendar import Calendar


class TestCalendar:
    """Test suite for Calendar module functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.cal = Calendar('2024-01-01', '2024-12-31')
        self.cal_2005 = Calendar('2005-01-01', '2005-12-31')
    
    def test_calendar_initialization(self):
        """Test calendar initialization with different date formats."""
        # Test string format
        cal1 = Calendar('2024-01-01', '2024-12-31')
        assert cal1.start_date.year == 2024
        assert cal1.end_date.year == 2024
        
        # Test integer format
        cal2 = Calendar(20240101, 20241231)
        assert cal2.start_date.year == 2024
        assert cal2.end_date.year == 2024
        
        # Test datetime format
        start_dt = datetime(2024, 1, 1)
        end_dt = datetime(2024, 12, 31)
        cal3 = Calendar(start_dt, end_dt)
        assert cal3.start_date.year == 2024
        assert cal3.end_date.year == 2024
    
    def test_date_validation_functions(self):
        """Test date validation functions (is_weekend, is_holiday, is_biz_day)."""
        # Test weekend detection
        assert self.cal.is_weekend('2024-01-06') == True  # Saturday
        assert self.cal.is_weekend('2024-01-07') == True  # Sunday
        assert self.cal.is_weekend('2024-01-08') == False  # Monday
        
        # Test holiday detection (weekends and non-weekend holidays)
        assert self.cal.is_holiday('2024-01-01') == True  # New Year's Day (20240101 in CSV)
        assert self.cal.is_holiday('2024-02-09') == True  # Spring Festival (20240209 in CSV)
        assert self.cal.is_holiday('2024-01-06') == True  # Saturday (weekend)
        assert self.cal.is_holiday('2024-01-07') == True  # Sunday (weekend)
        assert self.cal.is_holiday('2024-01-02') == False  # Not a holiday
        
        # Test non-weekend holiday detection specifically
        assert self.cal.is_non_weekend_holiday('2024-01-01') == True  # New Year's Day
        assert self.cal.is_non_weekend_holiday('2024-01-06') == False  # Saturday (weekend, not non-weekend holiday)
        assert self.cal.is_non_weekend_holiday('2024-01-02') == False  # Not a holiday
        
        # Test business day detection (not a holiday)
        assert self.cal.is_biz_day('2024-01-02') == True  # Tuesday, not holiday
        assert self.cal.is_biz_day('2024-01-06') == False  # Saturday (weekend holiday)
        assert self.cal.is_biz_day('2024-01-01') == False  # Non-weekend holiday
        
        # Test 2005 data
        assert self.cal_2005.is_holiday('2005-01-03') == True  # Holiday in CSV data
        assert self.cal_2005.is_holiday('2005-02-07') == True  # Holiday in CSV data
        assert self.cal_2005.is_biz_day('2005-01-03') == False  # Non-weekend holiday
    
    def test_date_list_queries(self):
        """Test holidays and biz_days methods."""
        # Test business days query
        biz_days = self.cal.biz_days('2024-01-01', '2024-01-05')
        assert len(biz_days) > 0
        assert isinstance(biz_days[0], int)  # Default format is integer
        
        # Test holidays query
        holidays = self.cal.holidays('2024-01-01', '2024-01-31')
        assert len(holidays) > 0
        
        # Test with include_weekends parameter
        holidays_no_weekends = self.cal.holidays('2024-01-01', '2024-01-31', include_weekends=False)
        holidays_with_weekends = self.cal.holidays('2024-01-01', '2024-01-31', include_weekends=True)
        assert len(holidays_with_weekends) >= len(holidays_no_weekends)
    
    def test_output_formats(self):
        """Test different output formats (integer, string, datetime)."""
        start_date = '2024-01-02'  # Use a business day to ensure we get results
        end_date = '2024-01-05'
        
        # Test integer format (default)
        dates_int = self.cal.biz_days(start_date, end_date)
        assert len(dates_int) > 0
        assert isinstance(dates_int[0], int)
        assert dates_int[0] == 20240102
        
        # Test string format
        dates_str = self.cal.biz_days(start_date, end_date, return_string=True)
        assert len(dates_str) > 0
        assert isinstance(dates_str[0], str)
        assert dates_str[0] == '2024-01-02'
        
        # Test datetime format
        dates_dt = self.cal.biz_days(start_date, end_date, return_dt=True)
        assert len(dates_dt) > 0
        assert hasattr(dates_dt[0], 'year')
        assert dates_dt[0].year == 2024
        
        # Verify all formats have same length
        assert len(dates_int) == len(dates_str) == len(dates_dt)
    
    def test_date_adjustment(self):
        """Test adjust_date method with different conventions."""
        # Test Following convention (default)
        adjusted_forward = self.cal.adjust_date('2024-01-06', convention=0)  # Saturday
        # Should adjust to next business day (Monday)
        assert self.cal.is_biz_day(adjusted_forward)
        
        # Test Preceding convention
        adjusted_backward = self.cal.adjust_date('2024-01-07', convention=2)  # Sunday
        # Should adjust to previous business day (Friday)
        assert self.cal.is_biz_day(adjusted_backward)
        
        # Test with already business day (should return same date)
        biz_date = '2024-01-02'  # Tuesday, not a holiday
        adjusted_same = self.cal.adjust_date(biz_date, return_string=True)
        assert adjusted_same == biz_date
        
        # Test invalid convention
        with pytest.raises(ValueError):
            self.cal.adjust_date('2024-01-01', convention=99)
    
    def test_date_advancement(self):
        """Test advance_date method with different periods."""
        base_date = '2024-01-02'  # Use a business day
        
        # Test business days
        advanced_5b = self.cal.advance_date(base_date, '5b', return_string=True)
        assert advanced_5b != base_date
        
        # Test negative business days
        advanced_neg3b = self.cal.advance_date('2024-01-10', '-3b', return_string=True)
        assert advanced_neg3b < '2024-01-10'
        
        # Test weeks
        advanced_1w = self.cal.advance_date(base_date, '1w', return_string=True)
        assert advanced_1w == '2024-01-09'
        
        # Test months
        advanced_1m = self.cal.advance_date(base_date, '1m', return_string=True)
        assert advanced_1m == '2024-02-02'
        
        # Test negative months
        advanced_neg1m = self.cal.advance_date(base_date, '-1m', return_string=True)
        assert advanced_neg1m == '2023-12-02'
        
        # Test years
        advanced_1y = self.cal.advance_date(base_date, '1y', return_string=True)
        assert advanced_1y == '2025-01-02'
        
        # Test invalid period format
        with pytest.raises(ValueError):
            self.cal.advance_date(base_date, 'invalid')
    
    def test_period_parsing(self):
        """Test _parse_period method."""
        # Test valid formats
        assert self.cal._parse_period('5b') == (5, 'b')
        assert self.cal._parse_period('-3w') == (-3, 'w')
        assert self.cal._parse_period('1m') == (1, 'm')
        assert self.cal._parse_period('2y') == (2, 'y')
        
        # Test invalid formats
        with pytest.raises(ValueError):
            self.cal._parse_period('invalid')
        with pytest.raises(ValueError):
            self.cal._parse_period('5x')
    
    def test_schedule_generation(self):
        """Test schedule method with different parameters."""
        start_date = '2024-01-02'  # Use a business day
        end_date = '2024-01-31'
        
        # Test forward schedule generation
        schedule_forward = self.cal.schedule(start_date, end_date, '1w', 1)
        assert len(schedule_forward) > 0
        assert schedule_forward[0] <= schedule_forward[-1]  # Should be ascending
        
        # Test backward schedule generation
        schedule_backward = self.cal.schedule(start_date, end_date, '1w', 2)
        assert len(schedule_backward) > 0
        assert schedule_backward[0] <= schedule_backward[-1]  # Should still be ascending after processing
        
        # Test business day schedule
        schedule_biz = self.cal.schedule('2024-01-02', '2024-01-15', '3b', 1)
        assert len(schedule_biz) > 0
        
        # Test monthly schedule
        schedule_monthly = self.cal.schedule('2024-01-02', '2024-06-30', '1m', 1, return_string=True)
        expected_months = ['2024-01-02', '2024-02-02', '2024-03-02', '2024-04-02', '2024-05-02', '2024-06-02']
        assert schedule_monthly == expected_months
        
        # Test invalid date generation rule
        with pytest.raises(ValueError):
            self.cal.schedule(start_date, end_date, '1w', 99)
    
    def test_date_parsing(self):
        """Test _parse_date method with different input formats."""
        # Test string formats
        parsed1 = self.cal._parse_date('2024-01-01')
        assert parsed1.year == 2024 and parsed1.month == 1 and parsed1.day == 1
        
        parsed2 = self.cal._parse_date('20240101')
        assert parsed2.year == 2024 and parsed2.month == 1 and parsed2.day == 1
        
        parsed3 = self.cal._parse_date('2024/01/01')
        assert parsed3.year == 2024 and parsed3.month == 1 and parsed3.day == 1
        
        # Test integer format
        parsed4 = self.cal._parse_date(20240101)
        assert parsed4.year == 2024 and parsed4.month == 1 and parsed4.day == 1
        
        # Test datetime format
        dt = datetime(2024, 1, 1)
        parsed5 = self.cal._parse_date(dt)
        assert parsed5 == dt
        
        # Test invalid formats
        with pytest.raises(ValueError):
            self.cal._parse_date('invalid-date')
        
        with pytest.raises(ValueError):
            self.cal._parse_date(12.34)  # Float not supported
    
    def test_legacy_compatibility(self):
        """Test get_trading_days method for backward compatibility."""
        # Test with default parameters
        trading_days = self.cal.get_trading_days()
        assert len(trading_days) > 0
        assert isinstance(trading_days, pd.DatetimeIndex)
        
        # Test with specific date range
        trading_days_range = self.cal.get_trading_days('2024-01-01', '2024-01-31')
        assert len(trading_days_range) > 0
        assert isinstance(trading_days_range, pd.DatetimeIndex)
        
        # Test unified logic: get_trading_days now uses the same business day logic as biz_days
        # Both should exclude weekends and non-weekend holidays
        biz_days_equivalent = self.cal.biz_days('2024-01-01', '2024-01-31', return_dt=True)
        
        # Convert both to sets of date strings for comparison
        trading_days_set = set(d.strftime('%Y-%m-%d') for d in trading_days_range)
        biz_days_set = set(d.strftime('%Y-%m-%d') for d in biz_days_equivalent)
        
        # Both methods should now return the same results
        assert trading_days_set == biz_days_set, f"Trading days and business days should be identical. Difference: {trading_days_set.symmetric_difference(biz_days_set)}"
        
        # Verify return type compatibility - get_trading_days should still return pd.DatetimeIndex
        assert isinstance(trading_days_range, pd.DatetimeIndex)
        assert not isinstance(biz_days_equivalent, pd.DatetimeIndex)  # biz_days returns list
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test leap year handling
        leap_cal = Calendar('2024-01-01', '2024-12-31')  # 2024 is a leap year
        feb29_result = leap_cal.advance_date('2024-02-29', '1y', return_string=True)
        assert feb29_result == '2025-02-28'  # Should handle leap year edge case
        
        # Test month overflow (Jan 31 + 1 month)
        jan31_result = self.cal.advance_date('2024-01-31', '1m', return_string=True)
        assert jan31_result == '2024-02-29'  # Should handle month overflow (2024 is leap year)
        
        # Test same start and end date
        same_date_biz = self.cal.biz_days('2024-01-02', '2024-01-02')
        assert len(same_date_biz) == 1
        
        # Test weekend-only range
        weekend_holidays = self.cal.holidays('2024-01-06', '2024-01-07', include_weekends=True)
        assert len(weekend_holidays) == 2  # Saturday and Sunday
    
    def test_repr_method(self):
        """Test string representation of Calendar object."""
        repr_str = repr(self.cal)
        assert 'Calendar' in repr_str
        assert '2024-01-01' in repr_str
        assert '2024-12-31' in repr_str
    
    def test_class_variable_sharing(self):
        """Test that non_weekend_holidays is shared across instances."""
        cal1 = Calendar('2024-01-01', '2024-12-31')
        cal2 = Calendar('2023-01-01', '2023-12-31')
        
        # Both instances should share the same non_weekend_holidays reference
        assert cal1.non_weekend_holidays is cal2.non_weekend_holidays
        assert Calendar.non_weekend_holidays is not None
        assert len(Calendar.non_weekend_holidays) > 0


def test_calendar_integration():
    """Integration test demonstrating typical usage patterns."""
    print("=== Calendar Integration Test ===")
    
    # Create calendar instance
    cal = Calendar('2024-01-01', '2024-12-31')
    print(f"Calendar created: {cal}")
    
    # Typical usage scenario 1: Find next business day
    today = '2024-01-05'  # Friday
    next_biz_day = cal.advance_date(today, '1b', return_string=True)
    print(f"Next business day after {today}: {next_biz_day}")
    
    # Typical usage scenario 2: Get month-end business day
    month_end = '2024-01-31'
    last_biz_day = cal.adjust_date(month_end, convention=2, return_string=True)
    print(f"Last business day of January 2024: {last_biz_day}")
    
    # Typical usage scenario 3: Generate quarterly schedule
    quarterly_dates = cal.schedule('2024-01-02', '2024-12-31', '3m', 1, return_string=True)
    print(f"Quarterly schedule: {quarterly_dates}")
    
    # Typical usage scenario 4: Count business days in a month
    jan_biz_days = cal.biz_days('2024-01-01', '2024-01-31')
    print(f"Business days in January 2024: {len(jan_biz_days)}")
    
    print("=== Integration test completed successfully ===")


if __name__ == "__main__":
    # Run integration test when script is executed directly
    test_calendar_integration()
