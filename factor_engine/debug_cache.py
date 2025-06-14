#!/usr/bin/env python3
"""
调试脚本：分析test_cache.py的死锁问题
用于详细跟踪执行流程，识别可能的死锁原因
"""

import sys
import time
import threading
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from unittest.mock import MagicMock

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


def debug_test_basic():
    """调试基本的缓存测试"""
    print("=== 开始基本调试测试 ===")
    
    # 1. 测试解析器
    print("1. 测试解析器...")
    parser = ExpressionParser()
    try:
        ast = parser.parse("ts_mean(close, 5)")
        print(f"   解析成功: {ast}")
    except Exception as e:
        print(f"   解析失败: {e}")
        return False
    
    # 2. 测试注册表
    print("2. 测试操作符注册表...")
    from factor_engine.registry import op_registry
    try:
        op = op_registry.get("ts_mean", window=5)
        print(f"   操作符获取成功: {op}")
    except Exception as e:
        print(f"   操作符获取失败: {e}")
        return False
    
    # 3. 测试规划器
    print("3. 测试规划器...")
    cache = InMemoryCache(max_size=10)
    context = ExecutionContext("2023-01-01", "2023-01-10", stocks=['AAPL', 'GOOG'])
    planner = ExecutionPlanner(cache=cache)
    
    try:
        dag = planner.plan(ast, context)
        print(f"   规划成功: {dag}")
        print(f"   DAG节点数: {len(planner.get_all_nodes())}")
        
        # 打印所有节点信息
        for node in planner.get_all_nodes():
            print(f"   节点: {node.expression}, 操作符: {node.operator}, 状态: {node.status}")
    except Exception as e:
        print(f"   规划失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 4. 测试调度器（同步模式）
    print("4. 测试调度器（同步模式）...")
    provider = create_mock_provider()
    scheduler = Scheduler(provider, cache=cache, max_workers=1, verbose=True)
    
    try:
        print("   开始执行...")
        start_time = time.time()
        result = scheduler.execute(dag, context)
        end_time = time.time()
        print(f"   执行成功，耗时: {end_time - start_time:.4f}s")
        print(f"   结果类型: {type(result)}")
        print(f"   数据提供者调用次数: {provider.load_call_count}")
        return True
    except Exception as e:
        print(f"   执行失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def debug_test_with_timeout():
    """带超时的调试测试"""
    print("\n=== 开始带超时的调试测试 ===")
    
    def timeout_handler():
        print("   [TIMEOUT] 10秒超时触发，可能存在死锁")
        # 打印所有线程的堆栈信息
        import faulthandler
        faulthandler.dump_traceback()
        
    # 设置10秒超时
    timer = threading.Timer(10.0, timeout_handler)
    timer.start()
    
    try:
        success = debug_test_basic()
        timer.cancel()
        return success
    except Exception as e:
        timer.cancel()
        print(f"   测试异常: {e}")
        import traceback
        traceback.print_exc()
        return False


def debug_parallel_mode():
    """调试并行模式"""
    print("\n=== 开始并行模式调试测试 ===")
    
    expression = "ts_mean(close, 5)"
    parser = ExpressionParser()
    cache = InMemoryCache(max_size=10)
    context = ExecutionContext("2023-01-01", "2023-01-10", stocks=['AAPL', 'GOOG'])
    planner = ExecutionPlanner(cache=cache)
    provider = create_mock_provider()
    
    # 测试并行调度器
    scheduler = Scheduler(provider, cache=cache, max_workers=2, verbose=True)
    
    try:
        ast = parser.parse(expression)
        dag = planner.plan(ast, context)
        
        print("   开始并行执行...")
        start_time = time.time()
        result = scheduler.execute(dag, context, timeout=5.0)  # 5秒超时
        end_time = time.time()
        
        print(f"   并行执行成功，耗时: {end_time - start_time:.4f}s")
        return True
    except Exception as e:
        print(f"   并行执行失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("开始调试 test_cache.py 的死锁问题...")
    
    # 首先导入operators模块确保所有操作符都注册
    print("导入操作符...")
    try:
        import factor_engine.operators
        print("操作符导入成功")
    except Exception as e:
        print(f"操作符导入失败: {e}")
        return
    
    # 测试基本功能
    if not debug_test_with_timeout():
        print("基本功能测试失败，停止进一步测试")
        return
    
    # 测试并行模式
    if not debug_parallel_mode():
        print("并行模式测试失败")
        return
    
    print("\n=== 所有调试测试完成 ===")


if __name__ == "__main__":
    main() 