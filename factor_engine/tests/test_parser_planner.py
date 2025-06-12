import pytest
from factor_engine.parser_planner.parser import ExpressionParser
from factor_engine.parser_planner.planner import ExecutionPlanner
from factor_engine.parser_planner.ast import OperatorNode, LeafNode

@pytest.fixture
def parser():
    return ExpressionParser()

@pytest.fixture
def planner():
    return ExecutionPlanner()

class TestParserPlanner:

    def test_parser_simple(self, parser):
        """测试解析器能否处理简单的表达式。"""
        expression = "ts_mean(close, 10)"
        ast = parser.parse(expression)
        
        assert isinstance(ast, OperatorNode)
        assert ast.value == "ts_mean"
        assert len(ast.args) == 2
        
        assert isinstance(ast.args[0], LeafNode)
        assert ast.args[0].value == "close"
        
        assert isinstance(ast.args[1], LeafNode)
        assert ast.args[1].value == 10

    def test_parser_nested_cse(self, parser):
        """测试解析器能否正确处理用于CSE测试的嵌套表达式。"""
        expression = "add(rank(close), rank(close))"
        ast = parser.parse(expression)

        assert isinstance(ast, OperatorNode)
        assert ast.value == "add"
        assert len(ast.args) == 2
        
        # 检查两个子节点是否结构相同
        assert isinstance(ast.args[0], OperatorNode)
        assert ast.args[0].value == "rank"
        assert isinstance(ast.args[0].args[0], LeafNode)
        assert ast.args[0].args[0].value == "close"

        assert str(ast.args[0]) == "rank('close')"
        assert str(ast.args[1]) == "rank('close')"

    def test_planner_common_subexpression_elimination(self, parser, planner):
        """
        验证规划器是否能为 add(rank(close), rank(close)) 生成包含节点重用的正确DAG。
        """
        expression = "add(rank(close), rank(close))"
        
        # 1. 解析
        ast = parser.parse(expression)
        
        # 2. 规划
        root_node = planner.plan(ast)
        all_nodes = planner.get_all_nodes()
        
        # 3. 验证
        # 预期有3个唯一的节点:
        # - add(rank('close'), rank('close'))
        # - rank('close')
        # - 'close'
        assert len(all_nodes) == 3
        
        # 按表达式字符串查找节点，进行更详细的断言
        node_map = {node.expression: node for node in all_nodes}
        
        add_node = node_map.get("add(rank('close'), rank('close'))")
        rank_node = node_map.get("rank('close')")
        close_node = node_map.get("'close'")
        
        assert add_node is not None
        assert rank_node is not None
        assert close_node is not None
        
        # 检查根节点
        assert root_node == add_node
        assert root_node.operator == "op_add"
        
        # 检查依赖关系
        # add 节点应该依赖于 rank 节点
        assert len(add_node.dependencies) == 1
        assert add_node.dependencies[0] == rank_node
        
        # rank 节点应该依赖于 close 节点
        assert len(rank_node.dependencies) == 1
        assert rank_node.dependencies[0] == close_node
        
        # close 节点是叶子节点，没有依赖
        assert len(close_node.dependencies) == 0
        assert close_node.operator == "load_data"
        
        # 检查 add 节点的参数
        # 两个参数都应该是同一个 rank_node 对象
        assert len(add_node.args) == 2
        assert add_node.args[0] is rank_node
        assert add_node.args[1] is rank_node 