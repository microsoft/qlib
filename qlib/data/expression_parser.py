"""Safe parser for Qlib's user-facing feature expression language."""

import ast
import operator

from qlib.utils import parse_field

from .base import Expression
from .ops import Operators


class ExpressionSyntaxError(ValueError):
    """Raised when an expression contains syntax outside Qlib's language."""


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.BitAnd: operator.and_,
    ast.BitOr: operator.or_,
}

_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
    ast.Invert: operator.invert,
}

_COMPARISON_OPERATORS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}


def _unsupported(node):
    raise ExpressionSyntaxError(f"Unsupported syntax in Qlib expression: {type(node).__name__}")


def _evaluate(node):
    if isinstance(node, ast.Constant):
        return node.value

    if isinstance(node, ast.List):
        return [_evaluate(item) for item in node.elts]

    if isinstance(node, ast.Tuple):
        return tuple(_evaluate(item) for item in node.elts)

    if isinstance(node, ast.BinOp):
        operation = _BINARY_OPERATORS.get(type(node.op))
        if operation is None:
            _unsupported(node.op)
        left = _evaluate(node.left)
        right = _evaluate(node.right)
        if not isinstance(left, Expression) and not isinstance(right, Expression):
            raise ExpressionSyntaxError("Constant-only arithmetic is not supported in Qlib expressions")
        return operation(left, right)

    if isinstance(node, ast.UnaryOp):
        operation = _UNARY_OPERATORS.get(type(node.op))
        if operation is None:
            _unsupported(node.op)
        return operation(_evaluate(node.operand))

    if isinstance(node, ast.Compare):
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise ExpressionSyntaxError("Chained comparisons are not supported in Qlib expressions")
        operation = _COMPARISON_OPERATORS.get(type(node.ops[0]))
        if operation is None:
            _unsupported(node.ops[0])
        left = _evaluate(node.left)
        right = _evaluate(node.comparators[0])
        if not isinstance(left, Expression) and not isinstance(right, Expression):
            raise ExpressionSyntaxError("Constant-only comparisons are not supported in Qlib expressions")
        return operation(left, right)

    if isinstance(node, ast.Call):
        if not (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "Operators"
        ):
            raise ExpressionSyntaxError("Only registered Qlib operators may be called")
        if node.func.attr.startswith("_"):
            raise ExpressionSyntaxError("Private operator names are not allowed")
        if any(isinstance(arg, ast.Starred) for arg in node.args):
            raise ExpressionSyntaxError("Starred arguments are not allowed")
        if any(keyword.arg is None for keyword in node.keywords):
            raise ExpressionSyntaxError("Expanded keyword arguments are not allowed")

        try:
            operation = getattr(Operators, node.func.attr)
        except AttributeError as exc:
            raise ExpressionSyntaxError(f"Unknown Qlib operator: {node.func.attr}") from exc

        args = [_evaluate(arg) for arg in node.args]
        kwargs = {keyword.arg: _evaluate(keyword.value) for keyword in node.keywords}
        return operation(*args, **kwargs)

    _unsupported(node)


def parse_expression(field) -> Expression:
    """Parse a Qlib expression without executing arbitrary Python code."""
    if isinstance(field, Expression):
        return field

    source = parse_field(field)
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError as exc:
        raise ExpressionSyntaxError(f"Invalid Qlib expression syntax: {field!r}") from exc

    expression = _evaluate(tree.body)
    if not isinstance(expression, Expression):
        raise ExpressionSyntaxError("A Qlib expression must produce an Expression object")
    return expression
