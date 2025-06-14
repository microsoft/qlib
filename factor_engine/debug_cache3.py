#!/usr/bin/env python3
"""
调试缓存写入问题
"""

import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from unittest.mock import MagicMock

# 导入操作符模块
import factor_engine.operators

from factor_engine.engine.cache import InMemoryCache
from factor_engine.parser_planner.parser import ExpressionParser
from factor_engine.parser_planner.planner import ExecutionPlanner
from factor_engine.engine.scheduler import Scheduler
from factor_engine.engine.context import ExecutionContext
from factor_engine.data_layer.loader import DataProvider
from factor_engine.data_layer.containers import PanelContainer
from factor_engine.core.dag import NodeStatus
from factor_engine.engine.utils import generate_cache_key


def create_mock_data():
    """创建测试数据"""
    dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'])
    stocks = ['AAPL', 'GOOG', 'MSFT']
    data = pd.DataFrame({
        'AAPL': [150, 152, 151, 155, 154],
        'GOOG': [2800, 2820, 2810, 2850, 2840],
        'MSFT': [300, 305, 302, 310, 308]
    }, index=dates)
    return PanelContainer(data)


def create_mock_provider():
    """创建带计数器的模拟数据提供者"""
    provider = MagicMock(spec=DataProvider)
    provider.load_call_count = 0
    sample_data = create_mock_data()

    def mock_load(field, **kwargs):
        print(f"[MockProvider] Loading field: {field}")
        provider.load_call_count += 1
        time.sleep(0.05)  # 模拟 I/O 延迟
        return PanelContainer(sample_data.get_data().copy())

    provider.load.side_effect = mock_load
    return provider


def test_cache_writing():
    """测试缓存写入"""
    print("=== 测试缓存写入 ===")
    
    expression = "ts_mean(close, 5)"
    parser = ExpressionParser()
    cache = InMemoryCache(max_size=10)
    context = ExecutionContext("2023-01-01", "2023-01-10", stocks=['AAPL', 'GOOG'])
    provider = create_mock_provider()

    planner = ExecutionPlanner(cache=cache)
    scheduler = Scheduler(provider, cache=cache, verbose=True)
    
    ast = parser.parse(expression)
    dag = planner.plan(ast, context)
    
    print(f"执行前缓存大小: {len(cache)}")
    
    # 手动检查缓存键
    cache_key = generate_cache_key(expression, context)
    print(f"预期的缓存键: {cache_key}")
    
    result = scheduler.execute(dag, context)
    
    print(f"执行后缓存大小: {len(cache)}")
    print("缓存键:")
    for key in cache._cache.keys():
        print(f"  - {key}")
        print(f"    值类型: {type(cache._cache[key])}")
    
    # 手动测试缓存写入
    print("\n=== 手动测试缓存写入 ===")
    test_key = "test_key"
    test_value = "test_value"
    cache.set(test_key, test_value)
    print(f"手动写入后缓存大小: {len(cache)}")
    retrieved = cache.get(test_key)
    print(f"检索到的值: {retrieved}")


if __name__ == "__main__":
    test_cache_writing() 