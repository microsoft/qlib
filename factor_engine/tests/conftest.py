import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
from factor_engine.data_layer.containers import PanelContainer

# 将项目根目录 (factor_engine) 添加到 Python 搜索路径
# 这能确保测试在运行时可以找到 'data_layer' 等模块
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

@pytest.fixture(scope="module")
def sample_panel_data_1() -> PanelContainer:
    """一个固定的 PanelContainer 实例，用于测试。"""
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
    """第二个固定的 PanelContainer 实例，用于二元运算测试。"""
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
    """一个包含 NaN 值的 PanelContainer。"""
    dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03'])
    stocks = ['AAPL', 'GOOG']
    data = np.array([
        [1, 2],
        [np.nan, 4],
        [3, np.nan],
    ])
    df = pd.DataFrame(data, index=dates, columns=stocks)
    return PanelContainer(df) 