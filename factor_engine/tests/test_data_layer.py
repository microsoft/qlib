import pytest
import pandas as pd
from pathlib import Path
import pyarrow

from data_layer.calendar import Calendar
from data_layer.loader import ParquetDataProvider
from data_layer.containers import PanelContainer

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