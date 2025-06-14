import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

# This ensures that when pytest runs, it can find the 'factor_engine' module.
# It adds the project's root directory (the parent of 'factor_engine') to the path.
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Imports from our module must come AFTER the sys.path modification
from factor_engine.data_layer.containers import PanelContainer
from factor_engine.data_layer.calendar import Calendar
from factor_engine.data_layer.loader import ParquetDataProvider


@pytest.fixture(scope="module")
def test_data_path():
    """Returns the root directory for test data."""
    return Path(__file__).parent / 'test_data'

@pytest.fixture(scope="module")
def calendar_fixture():
    """Provides a calendar instance for the tests."""
    # This range covers the dates in the test parquet files
    return Calendar(start_date="2023-01-01", end_date="2023-01-31")

@pytest.fixture(scope="module")
def data_provider_fixture(test_data_path, calendar_fixture):
    """Provides a ParquetDataProvider instance."""
    return ParquetDataProvider(data_path=str(test_data_path), calendar=calendar_fixture)

@pytest.fixture(scope="module")
def sample_panel_data_1() -> PanelContainer:
    """A fixed PanelContainer instance for testing."""
    dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'])
    stocks = ['AAPL', 'GOOG', 'MSFT']
    data = np.array([
        [150, 2800, 300],
        [152, 2820, 305],
        [151, 2810, 302],
        [155, 2850, 310],
        [154, 2840, 308],
    ], dtype=float)
    df = pd.DataFrame(data, index=dates, columns=stocks)
    return PanelContainer(df)

@pytest.fixture(scope="module")
def sample_panel_data_2() -> PanelContainer:
    """A second fixed PanelContainer instance for binary operation tests."""
    dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'])
    stocks = ['AAPL', 'GOOG', 'MSFT']
    data = np.array([
        [10, 20, 30],
        [12, 22, 35],
        [11, 21, 32],
        [15, 25, 40],
        [14, 24, 38],
    ], dtype=float)
    df = pd.DataFrame(data, index=dates, columns=stocks)
    return PanelContainer(df)

@pytest.fixture(scope="module")
def panel_with_nan() -> PanelContainer:
    """A PanelContainer with NaN values."""
    dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03'])
    stocks = ['AAPL', 'GOOG']
    data = np.array([
        [1, 2],
        [np.nan, 4],
        [3, np.nan],
    ])
    df = pd.DataFrame(data, index=dates, columns=stocks)
    return PanelContainer(df) 