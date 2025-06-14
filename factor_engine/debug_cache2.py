#!/usr/bin/env python3
"""
详细调试第二次执行的缓存问题
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


def test_detailed_cache():
    """详细的缓存测试"""
    print("=== 开始详细缓存测试 ===")
    
    expression = "ts_mean(close, 5)"
    parser = ExpressionParser()
    cache = InMemoryCache(max_size=10)
    context = ExecutionContext("2023-01-01", "2023-01-10", stocks=['AAPL', 'GOOG'])
    provider = create_mock_provider()

    # --- 第一次运行 ---
    print("\n--- 第一次运行 ---")
    planner1 = ExecutionPlanner(cache=cache)
    scheduler1 = Scheduler(provider, cache=cache, verbose=True)
    
    ast1 = parser.parse(expression)
    print(f"AST1: {ast1}")
    
    dag1 = planner1.plan(ast1, context)
    print(f"DAG1 根节点: {dag1.expression}, 状态: {dag1.status}")
    print(f"DAG1 所有节点:")
    for node in planner1.get_all_nodes():
        print(f"  - {node.expression}, 操作符: {node.operator}, 状态: {node.status}")

    start_time1 = time.time()
    result1 = scheduler1.execute(dag1, context)
    duration1 = time.time() - start_time1
    
    print(f"第一次执行完成，耗时: {duration1:.4f}s")
    print(f"结果类型: {type(result1)}")
    print(f"数据提供者调用次数: {provider.load_call_count}")
    print(f"缓存内容数量: {len(cache)}")
    
    # 检查缓存内容
    print("缓存键:")
    for key in cache._cache.keys():
        print(f"  - {key}")

    # --- 第二次运行 ---
    print("\n--- 第二次运行 ---")
    provider.load_call_count = 0
    
    # 重用 planner1 （这很重要，因为它包含了缓存状态）
    scheduler2 = Scheduler(provider, cache=cache, verbose=True)
    
    ast2 = parser.parse(expression)
    print(f"AST2: {ast2}")
    
    dag2 = planner1.plan(ast2, context)
    print(f"DAG2 根节点: {dag2.expression}, 状态: {dag2.status}")
    print(f"DAG2 所有节点:")
    for node in planner1.get_all_nodes():
        print(f"  - {node.expression}, 操作符: {node.operator}, 状态: {node.status}")
    print(f"DAG2 根节点依赖数量: {len(dag2.dependencies)}")
    
    # 检查根节点的result_ref
    if dag2.status == NodeStatus.CACHED:
        print(f"根节点已缓存，result_ref类型: {type(dag2.result_ref)}")
        if dag2.result_ref:
            print(f"result_ref值: {dag2.result_ref}")
        else:
            print("警告: result_ref是None!")

    start_time2 = time.time()
    result2 = scheduler2.execute(dag2, context)
    duration2 = time.time() - start_time2

    print(f"第二次执行完成，耗时: {duration2:.4f}s")
    print(f"结果类型: {type(result2)}")
    print(f"数据提供者调用次数: {provider.load_call_count}")

    return result1, result2


if __name__ == "__main__":
    test_detailed_cache() 