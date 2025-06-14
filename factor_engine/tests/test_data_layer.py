import pytest
import pandas as pd
from pathlib import Path
import pyarrow
import unittest

from factor_engine.data_layer.calendar import Calendar
from factor_engine.data_layer.loader import ParquetDataProvider, load_daily_bundle
from factor_engine.data_layer.containers import PanelContainer

@pytest.fixture(scope="module")
def test_data_path():
    """返回测试数据的根目录。"""
    return Path(__file__).parent / 'test_data'

@pytest.fixture(scope="module")
def calendar_fixture():
    """提供一个贯穿测试的日历实例。"""
    return Calendar(start_date="2023-01-01", end_date="2023-01-31")

@pytest.fixture(scope="module")
def data_provider_fixture(test_data_path, calendar_fixture):
    """提供一个 ParquetDataProvider 实例。"""
    return ParquetDataProvider(data_path=str(test_data_path), calendar=calendar_fixture)

def test_load_single_field_all_stocks(data_provider_fixture):
    """测试加载单个字段的全量股票数据。"""
    start_date = "2023-01-03"
    end_date = "2023-01-05"
    
    container = data_provider_fixture.load(field="close", start_date=start_date, end_date=end_date)
    
    assert isinstance(container, PanelContainer)
    df = container.get_data()
    
    assert df.shape == (3, 2)
    assert not df.isnull().values.any()
    assert 'SH600000' in df.columns and 'SZ000001' in df.columns
    assert pd.to_datetime(start_date) in df.index and pd.to_datetime(end_date) in df.index
    assert df.loc[pd.to_datetime('2023-01-03'), 'SH600000'] == 10.1

def test_load_with_stock_selection(data_provider_fixture):
    """测试加载数据时筛选部分股票。"""
    start_date = "2023-01-03"
    end_date = "2023-01-05"
    stocks = ["SZ000001"]
    
    container = data_provider_fixture.load(field="volume", start_date=start_date, end_date=end_date, stocks=stocks)
    df = container.get_data()

    assert df.shape == (3, 1)
    assert 'SZ000001' in df.columns
    assert 'SH600000' not in df.columns
    assert df.loc[pd.to_datetime('2023-01-04'), 'SZ000001'] == 2100

def test_load_non_existent_field(data_provider_fixture):
    """测试加载一个不存在的字段时应如何处理。"""
    # pyarrow 会在读取不存在的列时抛出它自己的异常
    with pytest.raises(pyarrow.lib.ArrowInvalid):
        data_provider_fixture.load(field="non_existent_field", start_date="2023-01-03", end_date="2023-01-05")

def test_panel_container_methods(data_provider_fixture):
    """测试 PanelContainer 的辅助方法是否正常工作。"""
    container = data_provider_fixture.load(field="close", start_date="2023-01-03", end_date="2023-01-05")
    
    assert container.get_shape() == (3, 2)
    assert all(d in container.get_dates() for d in pd.to_datetime(["2023-01-03", "2023-01-04", "2023-01-05"]))
    assert "SH600000" in container.get_stocks()
    
    stock_series = container.get_stock_data("SH600000")
    assert isinstance(stock_series, pd.Series)
    assert len(stock_series) == 3
    
    date_series = container.get_date_data(pd.to_datetime("2023-01-05"))
    assert isinstance(date_series, pd.Series)
    assert date_series['SZ000001'] == 20.3

# Using unittest.TestCase for more complex assertions within the pytest framework
class TestLoadDailyBundle(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Set up data path once for all tests in this class."""
        # This approach avoids using pytest fixtures directly in a unittest.TestCase
        # A bit of a workaround to get the path.
        cls.db_path = Path(__file__).parent.parent.parent / "database"

    def test_load_all_data(self):
        """Test loading all data without any filters."""
        df = load_daily_bundle(data_path=self.db_path)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)
        self.assertEqual(df.index.names, ["date", "instrument"])
        self.assertIsInstance(df.index.get_level_values('date')[0], pd.Timestamp)
        # check if all expected columns are loaded
        self.assertTrue(all(c in df.columns for c in ['open', 'high', 'low', 'close', 'volume', 'turnover']))

    def test_load_with_time_range(self):
        """Test loading data for a specific time range."""
        start_time = "2024-01-02"
        end_time = "2024-01-04"
        df = load_daily_bundle(data_path=self.db_path, start_time=start_time, end_time=end_time)
        self.assertFalse(df.empty)
        dates = df.index.get_level_values('date').unique()
        self.assertGreaterEqual(dates.min(), pd.to_datetime(start_time))
        self.assertLessEqual(dates.max(), pd.to_datetime(end_time))
        # Assuming there are 3 trading days in this range in the test data
        self.assertEqual(len(dates), 3)

    def test_load_with_instruments(self):
        """Test loading data for specific instruments."""
        instruments = ["000001.SZ", "000002.SZ"]
        df = load_daily_bundle(data_path=self.db_path, instruments=instruments)
        self.assertFalse(df.empty)
        loaded_instruments = df.index.get_level_values('instrument').unique()
        self.assertCountEqual(loaded_instruments, instruments)

    def test_load_with_single_instrument(self):
        """Test loading data for a single instrument passed as a string."""
        instrument = "000001.SZ"
        df = load_daily_bundle(data_path=self.db_path, instruments=[instrument])
        self.assertFalse(df.empty)
        loaded_instruments = df.index.get_level_values('instrument').unique()
        self.assertEqual(len(loaded_instruments), 1)
        self.assertEqual(loaded_instruments[0], instrument)

    def test_load_no_data_for_time_range(self):
        """Test loading data for a time range with no data files."""
        start_time = "1999-01-01"
        end_time = "1999-01-05"
        df = load_daily_bundle(data_path=self.db_path, start_time=start_time, end_time=end_time)
        self.assertTrue(df.empty)
    
    def test_invalid_date_format(self):
        """Test that an invalid date format raises a ValueError."""
        with self.assertRaises(ValueError):
            load_daily_bundle(data_path=self.db_path, start_time="invalid-date") 