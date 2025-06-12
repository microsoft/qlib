from typing import Dict, List, Any

from factor_engine.core.dag import DAGNode
from .ast import ASTNode, OperatorNode, LeafNode

class ExecutionPlanner:
    """
    遍历 AST 并构建一个优化的执行 DAG。
    
    核心功能:
    - 将 AST 节点转换为 DAG 节点。
    - 实现公共子表达式消除 (CSE)。
    """
    def __init__(self):
        # 用于存储已创建的 DAG 节点，实现 CSE
        # Key: 节点的规范化表达式字符串, e.g., "rank(close)"
        # Value: 对应的 DAGNode 对象
        self._nodes_map: Dict[str, DAGNode] = {}

    def plan(self, ast_node: ASTNode) -> DAGNode:
        """
        将一个 AST 树转换为一个 DAG。
        这是一个递归函数，自顶向下遍历 AST 并构建 DAG 节点。
        """
        expression_str = str(ast_node)
        
        if expression_str in self._nodes_map:
            return self._nodes_map[expression_str]
        
        new_node: DAGNode

        if isinstance(ast_node, OperatorNode):
            # 这是一个操作节点 (e.g., add, rank)
            child_dag_nodes = [self.plan(arg) for arg in ast_node.args]
            
            op_args = []
            dependencies = []
            for child_node in child_dag_nodes:
                # 'literal' 类型的节点代表一个简单的值（如数字），而不是一个可执行的任务
                # 因此它不是一个依赖项，我们只关心它的值。
                if child_node.operator == 'literal':
                    op_args.append(child_node.args[0])
                else:
                    op_args.append(child_node)
                    dependencies.append(child_node)
            
            new_node = DAGNode(
                expression=expression_str,
                operator=f"op_{ast_node.value}",
                args=op_args,
                kwargs={}
            )
            
            for dep in dependencies:
                new_node.add_dependency(dep)

        elif isinstance(ast_node, LeafNode):
            # 这是一个叶子节点 (e.g., 'close' or 10)
            # 我们需要区分它是需要加载的数据字段还是一个纯粹的字面量参数。
            is_field = isinstance(ast_node.value, str) and not ast_node.value.replace('.', '', 1).isdigit()
            
            if is_field:
                # 字符串被认为是需要加载的数据字段
                new_node = DAGNode(
                    expression=expression_str,
                    operator="load_data",
                    args=[ast_node.value],
                    kwargs={}
                )
            else:
                # 数字或数字字符串被认为是字面量。
                # 创建一个特殊的 "literal" 节点，以便父节点可以提取其值。
                new_node = DAGNode(
                    expression=expression_str,
                    operator='literal',
                    args=[ast_node.value],
                    kwargs={}
                )
        else:
            raise TypeError(f"未知的 AST 节点类型: {type(ast_node)}")

        self._nodes_map[expression_str] = new_node
        return new_node

    def get_all_nodes(self) -> List[DAGNode]:
        """返回所有创建的唯一节点。"""
        return list(self._nodes_map.values()) 