import pytest
import time
import pandas as pd
from unittest.mock import MagicMock

from factor_engine.parser_planner.parser import ExpressionParser
from factor_engine.parser_planner.planner import ExecutionPlanner
from factor_engine.engine.scheduler import Scheduler, ExecutionContext
from factor_engine.engine.utils import load_operators
from factor_engine.data_layer.loader import DataProvider
from factor_engine.data_layer.containers import PanelContainer


@pytest.fixture(scope="module")
def expression_parser():
    return ExpressionParser()

@pytest.fixture(scope="module")
def all_operators():
    """动态加载所有算子。"""
    return load_operators()

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

    def test_end_to_end_simple_expression(self, expression_parser, all_operators, mock_data_provider, sample_panel_data_1, sample_panel_data_2):
        """测试一个简单表达式的端到端流程: add(close, open)"""
        # 1. 准备
        expression = "add(close, open)"
        planner = ExecutionPlanner()
        scheduler = Scheduler(mock_data_provider, all_operators)
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

    def test_parallel_execution_speedup(self, expression_parser, all_operators, mock_data_provider, sample_panel_data_1):
        """
        通过手动构建一个 add(slow_op_1, slow_op_2) 的 DAG 来验证并行加速效果。
        由于两个 slow_op 数据加载是并行的，总耗时不应是它们单独耗时的简单相加。
        """
        planner = ExecutionPlanner()
        scheduler = Scheduler(mock_data_provider, all_operators, max_workers=2)
        context = ExecutionContext("2023-01-01", "2023-01-05")

        # 手动构建 AST 和 DAG
        ast1 = expression_parser.parse("slow_op_1")
        ast2 = expression_parser.parse("slow_op_2")
        node1 = planner.plan(ast1, context)
        node2 = planner.plan(ast2, context)
        
        # 创建根节点
        root_node = planner.plan(expression_parser.parse("add(slow_op_1, slow_op_2)"), context)

        # `slow_op` 在 mock 中会 sleep 0.1 秒
        # 两个并行执行，总时间应该略大于 0.1 秒，但远小于 0.2 秒
        start_time = time.time()
        result = scheduler.execute(root_node, context)
        end_time = time.time()
        
        duration = end_time - start_time
        print(f"并行执行耗时: {duration:.4f}s")

        assert duration < 0.18 # 0.1s + some overhead, must be less than serial execution (0.2s)
        
        # 验证结果的正确性
        expected_data = sample_panel_data_1.get_data() + sample_panel_data_1.get_data()
        pd.testing.assert_frame_equal(result.get_data(), expected_data)

    def test_execution_failure(self, expression_parser, all_operators, mock_data_provider):
        """测试当某个节点执行失败时，调度器是否能正确抛出异常。"""
        # "unknown_field" 将在 mock_data_provider 中引发 ValueError
        expression = "add(close, unknown_field)"
        planner = ExecutionPlanner()
        scheduler = Scheduler(mock_data_provider, all_operators)
        context = ExecutionContext("2023-01-01", "2023-01-05")

        ast = expression_parser.parse(expression)
        dag = planner.plan(ast, context)

        with pytest.raises(ValueError, match="未知的测试字段: unknown_field"):
            scheduler.execute(dag, context) 