import pytest
import time
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

# 使用 conftest.py 中的 fixtures

class TestInMemoryCache:
    def test_lru_eviction(self):
        """测试 InMemoryCache 的 LRU 回收策略是否正常工作。"""
        cache = InMemoryCache(max_size=2)
        cache.set('a', 1)
        cache.set('b', 2)
        assert 'a' in cache and 'b' in cache

        # 访问 'a'，使其成为最近使用的项
        cache.get('a')
        
        # 添加 'c'，应该会挤出最近最少使用的 'b'
        cache.set('c', 3)
        assert 'a' in cache
        assert 'c' in cache
        assert 'b' not in cache
        assert len(cache) == 2

class TestCacheIntegration:

    @pytest.fixture
    def mock_data_provider_with_counter(self, sample_panel_data_1):
        """
        一个带有调用计数器的模拟 DataProvider。
        每次调用 load 都会使计数器加一。
        """
        provider = MagicMock(spec=DataProvider)
        provider.load_call_count = 0

        def mock_load(field, **kwargs):
            provider.load_call_count += 1
            time.sleep(0.05) # 模拟 I/O 延迟
            return PanelContainer(sample_panel_data_1.get_data().copy())

        provider.load.side_effect = mock_load
        return provider

    def test_caching_and_short_circuit(self, mock_data_provider_with_counter):
        """
        测试端到端的缓存流程：
        1. 第一次运行表达式，正常计算并缓存结果。
        2. 第二次运行相同表达式，应从缓存加载，速度显著加快，且数据加载次数为 0。
        """
        expression = "ts_mean(close, 5)"
        parser = ExpressionParser()
        cache = InMemoryCache(max_size=10)
        context = ExecutionContext("2023-01-01", "2023-01-10", stocks=['AAPL', 'GOOG'])

        # --- 第一次运行 ---
        planner1 = ExecutionPlanner(cache=cache)
        scheduler1 = Scheduler(mock_data_provider_with_counter, cache=cache)
        
        ast1 = parser.parse(expression)
        dag1 = planner1.plan(ast1, context)

        start_time1 = time.time()
        result1 = scheduler1.execute(dag1, context)
        duration1 = time.time() - start_time1
        
        # 验证第一次运行是否正确
        assert mock_data_provider_with_counter.load_call_count == 1
        assert isinstance(result1, PanelContainer)

        # --- 第二次运行 ---
        mock_data_provider_with_counter.load_call_count = 0
        
        # 重用 planner1，但创建新的 scheduler
        scheduler2 = Scheduler(mock_data_provider_with_counter, cache=cache)
        
        ast2 = parser.parse(expression)
        dag2 = planner1.plan(ast2, context)
        
        # 验证 DAG 是否被短路
        root_node = dag2
        assert root_node.status == NodeStatus.CACHED
        assert len(root_node.dependencies) == 0

        start_time2 = time.time()
        result2 = scheduler2.execute(dag2, context)
        duration2 = time.time() - start_time2

        # 验证第二次运行是否从缓存加载
        assert mock_data_provider_with_counter.load_call_count == 0, "DataProvider 不应该被调用"
        
        # 验证结果是否一致
        pd.testing.assert_frame_equal(result1.get_data(), result2.get_data())
        
        # 验证性能
        print(f"第一次运行耗时: {duration1:.4f}s")
        print(f"第二次运行耗时 (有缓存): {duration2:.4f}s")
        assert duration2 < duration1, "有缓存的运行应该比没有缓存的快"
        assert duration2 < 0.01, "有缓存的运行应该非常快" 