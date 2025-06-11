import hashlib
from enum import Enum
from typing import Any, List, Dict

class NodeStatus(Enum):
    PENDING = "PENDING"
    READY = "READY"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    CACHED = "CACHED"
    FAILED = "FAILED"

class DAGNode:
    """
    表示计算图中一个节点的类。
    每个节点代表一个操作或一个数据源。
    """
    def __init__(self, expression: str, operator: str, args: List[Any], kwargs: Dict[str, Any]):
        # --- 规划阶段填充 (Immutable after planning) ---
        self.expression: str = expression  # 节点的规范化表达式字符串, e.g., "rank(close)"
        self.id: str = self._generate_id(expression)  # 基于规范化表达式的唯一哈希值
        
        self.operator: str = operator  # e.g., 'op_rank', 'load_data'
        self.args: List[Any] = args  # 操作的位置参数, 其中 DAGNode 类型的参数代表依赖
        self.kwargs: Dict[str, Any] = kwargs # 操作的关键字参数

        self.dependencies: List['DAGNode'] = []  # 子节点 (该节点依赖的节点)
        self.parents: List['DAGNode'] = []      # 父节点 (依赖于该节点的节点)
        
        # --- 元数据 (Filled during planning) ---
        self.metadata: Dict[str, Any] = {
            'required_window': 0,
            'data_type_hint': None,
        }

        # --- 执行阶段状态 (Mutable during execution) ---
        self.status: NodeStatus = NodeStatus.PENDING
        self.result_ref: Any = None  # 指向缓存或内存中结果的引用
        self.in_degree: int = 0      # 在图中的入度，用于拓扑排序

    @staticmethod
    def _generate_id(expression: str) -> str:
        """根据表达式字符串生成唯一的节点 ID。"""
        return hashlib.sha256(expression.encode('utf-8')).hexdigest()

    def add_dependency(self, node: 'DAGNode'):
        """添加一个子依赖节点，并更新父子关系。"""
        if node not in self.dependencies:
            self.dependencies.append(node)
            node.parents.append(self)
            
    def update_in_degree(self):
        """根据依赖项计算节点的入度。"""
        self.in_degree = len(self.dependencies)

    def __repr__(self) -> str:
        return f"DAGNode(id={self.id[:8]}..., expr='{self.expression}')"

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, DAGNode):
            return NotImplemented
        return self.id == other.id 