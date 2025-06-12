from dataclasses import dataclass
from typing import List, Any

@dataclass
class ASTNode:
    """Represents a node in the Abstract Syntax Tree."""
    node_type: str  # 'operator' or 'literal'
    value: Any
    args: List['ASTNode']

    def __repr__(self):
        if self.node_type == 'literal':
            return f"'{self.value}'"
        return f"{self.value}({', '.join(map(str, self.args))})"

class LeafNode(ASTNode):
    """Represents a leaf node in the AST, which is a literal value like a field name or a number."""
    def __init__(self, value: Any):
        super().__init__(node_type='literal', value=value, args=[])

class OperatorNode(ASTNode):
    """Represents an operator node in the AST."""
    def __init__(self, operator: str, args: List[ASTNode]):
        super().__init__(node_type='operator', value=operator, args=args) 