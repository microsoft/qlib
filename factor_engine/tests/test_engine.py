import pytest
import time
import pandas as pd
import factor_engine.operators
from unittest.mock import MagicMock

from factor_engine.parser_planner.parser import ExpressionParser
from factor_engine.parser_planner.planner import ExecutionPlanner
from factor_engine.engine.scheduler import Scheduler, ExecutionContext
from factor_engine.data_layer.loader import DataProvider
from factor_engine.data_layer.containers import PanelContainer


@pytest.fixture(scope="module")
def expression_parser():
    return ExpressionParser()

@pytest.fixture
def mock_data_provider(sample_panel_data_1, sample_panel_data_2):
    """一个模拟的 DataProvider，根据字段名返回不同的测试数据。"""
    provider = MagicMock(spec=DataProvider)
    
    def mock_load(field, **kwargs):
        # 模拟一个耗时操作
        if 'slow_op' in field:
            time.sleep(0.1)
            return PanelContainer(sample_panel_data_1.get_data().copy())
        if field == 'close':
            # 返回一个副本以避免测试间的状态污染
            return PanelContainer(sample_panel_data_1.get_data().copy())
        if field == 'open':
            return PanelContainer(sample_panel_data_2.get_data().copy())
        raise ValueError(f"未知的测试字段: {field}")
        
    provider.load.side_effect = mock_load
    return provider

class TestScheduler:

    def test_end_to_end_simple_expression(self, expression_parser, mock_data_provider, sample_panel_data_1, sample_panel_data_2):
        """测试一个简单表达式的端到端流程: add(close, open)"""
        # 1. 准备
        expression = "add(close, open)"
        planner = ExecutionPlanner()
        scheduler = Scheduler(mock_data_provider, verbose=True)
        context = ExecutionContext("2023-01-01", "2023-01-05")

        # 2. 规划
        ast = expression_parser.parse(expression)
        dag = planner.plan(ast, context)

        # 3. 执行
        result_container = scheduler.execute(dag, context)

        # 4. 验证
        assert isinstance(result_container, PanelContainer)
        expected_data = sample_panel_data_1.get_data() + sample_panel_data_2.get_data()
        pd.testing.assert_frame_equal(result_container.get_data(), expected_data)
        
        # 验证 data_provider 被正确调用
        assert mock_data_provider.load.call_count == 2
        mock_data_provider.load.assert_any_call(field='close', start_date=context.start_date, end_date=context.end_date, stocks=context.stocks)
        mock_data_provider.load.assert_any_call(field='open', start_date=context.start_date, end_date=context.end_date, stocks=context.stocks)

    def test_parallel_execution_speedup(self, expression_parser, mock_data_provider, sample_panel_data_1):
        """
        通过 add(slow_op_1, slow_op_2) 的 DAG 来验证并行加速效果。
        """
        planner = ExecutionPlanner()
        scheduler = Scheduler(mock_data_provider, max_workers=2, verbose=True)
        context = ExecutionContext("2023-01-01", "2023-01-05")

        # 创建根节点
        root_node = planner.plan(expression_parser.parse("add(slow_op_1, slow_op_2)"), context)

        # `slow_op` 在 mock 中会 sleep 0.1 秒
        # 两个并行执行，总时间应该略大于 0.1 秒，但远小于 0.2 秒
        start_time = time.time()
        result = scheduler.execute(root_node, context)
        end_time = time.time()
        
        duration = end_time - start_time
        print(f"并行执行耗时: {duration:.4f}s")

        assert 0.1 < duration < 0.18 # 0.1s + some overhead, must be less than serial execution (0.2s)
        
        # 验证结果的正确性
        expected_data = sample_panel_data_1.get_data() + sample_panel_data_1.get_data()
        pd.testing.assert_frame_equal(result.get_data(), expected_data)

    def test_synchronous_execution(self, expression_parser, mock_data_provider, sample_panel_data_1):
        """
        通过 max_workers=1 验证同步(串行)执行模式。
        """
        planner = ExecutionPlanner()
        # 使用 max_workers=1 强制进入同步模式
        scheduler = Scheduler(mock_data_provider, max_workers=1, verbose=True)
        context = ExecutionContext("2023-01-01", "2023-01-05")

        root_node = planner.plan(expression_parser.parse("add(slow_op_1, slow_op_2)"), context)

        # 两个 slow_op (每个0.1s) 串行执行，总时间应该大于 0.2s
        start_time = time.time()
        result = scheduler.execute(root_node, context)
        end_time = time.time()
        
        duration = end_time - start_time
        print(f"同步执行耗时: {duration:.4f}s")
        assert duration > 0.2

        # 验证结果
        expected_data = sample_panel_data_1.get_data() + sample_panel_data_1.get_data()
        pd.testing.assert_frame_equal(result.get_data(), expected_data)

    def test_execution_failure_in_sync_mode(self, expression_parser, mock_data_provider):
        """测试在同步模式下，当某个节点执行失败时，调度器是否能正确抛出异常。"""
        expression = "add(close, unknown_field)"
        planner = ExecutionPlanner()
        scheduler = Scheduler(mock_data_provider, max_workers=1, verbose=True)
        context = ExecutionContext("2023-01-01", "2023-01-05")

        ast = expression_parser.parse(expression)
        dag = planner.plan(ast, context)

        # 在同步模式下，根本错误会被包装在 RuntimeError 中
        with pytest.raises(RuntimeError) as excinfo:
            scheduler.execute(dag, context)
        
        # 检查根本原因是否是我们期望的 ValueError
        assert isinstance(excinfo.value.__cause__, ValueError)
        assert "未知的测试字段: unknown_field" in str(excinfo.value.__cause__)

    def test_execution_timeout(self, expression_parser, mock_data_provider):
        """测试并行执行的超时机制。"""
        expression = "slow_op_1"
        planner = ExecutionPlanner()
        scheduler = Scheduler(mock_data_provider, max_workers=2, verbose=True)
        context = ExecutionContext("2023-01-01", "2023-01-05")

        ast = expression_parser.parse(expression)
        dag = planner.plan(ast, context)

        # slow_op 需要 0.1 秒，我们设置一个更短的超时
        with pytest.raises(TimeoutError, match="Execution timed out"):
            scheduler.execute(dag, context, timeout=0.05) 