from typing import Dict, List, Any, Optional

from factor_engine.core.dag import DAGNode, NodeStatus
from factor_engine.engine.cache import Cache
from factor_engine.engine.scheduler import ExecutionContext
from factor_engine.engine.utils import generate_cache_key
from .ast import ASTNode, OperatorNode, LeafNode

class ExecutionPlanner:
    """
    遍历 AST 并构建一个优化的执行 DAG。
    
    核心功能:
    - 将 AST 节点转换为 DAG 节点。
    - 实现公共子表达式消除 (CSE)。
    - 实现缓存短路。
    """
    def __init__(self, cache: Optional[Cache] = None):
        """
        Args:
            cache: 用于检查缓存命中的缓存对象。
        """
        # 用于存储已创建的 DAG 节点，实现 CSE
        # Key: 节点的规范化表达式字符串, e.g., "rank(close)"
        # Value: 对应的 DAGNode 对象
        self._nodes_map: Dict[str, DAGNode] = {}
        self._cache = cache

    def plan(self, ast_node: ASTNode, context: Optional[ExecutionContext] = None) -> DAGNode:
        """
        将一个 AST 树转换为一个 DAG。
        这是一个递归函数，自顶向下遍历 AST 并构建 DAG 节点。
        """
        expression_str = str(ast_node)
        
        # 1. 优先检查公共子表达式
        if expression_str in self._nodes_map:
            existing_node = self._nodes_map[expression_str]
            # 如果这个节点已经被计算过(在同一次规划中)，我们可以认为它是一种 "in-memory" 缓存命中
            if existing_node.status in (NodeStatus.COMPLETED, NodeStatus.CACHED):
                 # 创建一个克隆的、状态为 CACHED 的节点以清晰地表示短路
                return self._create_cached_node(expression_str, existing_node.result_ref)
            return existing_node
        
        # 2. 检查外部缓存
        if self._cache and context:
            cache_key = generate_cache_key(expression_str, context)
            cached_result = self._cache.get(cache_key)
            if cached_result is not None:
                node = self._create_cached_node(expression_str, cached_result)
                self._nodes_map[expression_str] = node
                return node
        
        node = self._create_node(ast_node, context)
        self._nodes_map[expression_str] = node
        return node

    def _create_node(self, ast_node: ASTNode, context: Optional[ExecutionContext] = None) -> DAGNode:
        """根据 AST 节点创建新的 DAGNode (缓存未命中时)。"""
        expression_str = str(ast_node)
        
        if isinstance(ast_node, OperatorNode):
            dependencies = [self.plan(arg, context) for arg in ast_node.args]
            
            op_args, true_deps = [], []
            for dep in dependencies:
                if dep.operator == 'literal':
                    op_args.append(dep.args[0])
                else:
                    op_args.append(dep)
                    true_deps.append(dep)
            
            node = DAGNode(
                expression=expression_str,
                operator=f"op_{ast_node.value}",
                args=op_args,
                kwargs={}
            )
            for dep in true_deps:
                node.add_dependency(dep)
            return node

        elif isinstance(ast_node, LeafNode):
            is_field = isinstance(ast_node.value, str) and not ast_node.value.replace('.', '', 1).isdigit()
            node = DAGNode(
                expression=expression_str,
                operator="load_data" if is_field else "literal",
                args=[ast_node.value],
                kwargs={}
            )
            return node
        else:
            raise TypeError(f"未知的 AST 节点类型: {type(ast_node)}")

    def _create_cached_node(self, expression: str, result: Any) -> DAGNode:
        """创建一个代表缓存命中结果的特殊 DAGNode。"""
        # 对于缓存的节点，我们不需要知道它的具体 operator 或 args，
        # 因为它永远不会被执行。
        node = DAGNode(
            expression=expression,
            operator='cached',
            args=[],
            kwargs={}
        )
        node.status = NodeStatus.CACHED
        node.result_ref = result
        return node

    def get_all_nodes(self) -> List[DAGNode]:
        """返回所有创建的唯一节点。"""
        return list(self._nodes_map.values()) 