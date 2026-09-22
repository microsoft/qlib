"""Safe parser for Qlib's user-facing feature expression language."""

import ast
import operator
import tokenize

from qlib.utils import parse_field
from qlib.utils.mod import CONFIG_MIGRATION_GUIDE

from .base import Expression
from .ops import Operators


class ExpressionSyntaxError(ValueError):
    """Raised when an expression contains syntax outside Qlib's language."""

    def __str__(self):
        return f"{super().__str__()}. Migration guide: {CONFIG_MIGRATION_GUIDE}"


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

_SCALAR_TYPES = (int, float, bool, str, bytes, type(None))
_MAX_CONTAINER_ITEMS = 4096
_SUPPORTED_NODES = (
    (
        ast.Expression,
        ast.Constant,
        ast.List,
        ast.Tuple,
        ast.Dict,
        ast.BinOp,
        ast.UnaryOp,
        ast.Compare,
        ast.BoolOp,
        ast.IfExp,
        ast.Subscript,
        ast.Index,
        ast.Slice,
        ast.Starred,
        ast.keyword,
        ast.Load,
        ast.And,
        ast.Or,
        ast.Not,
    )
    + tuple(_BINARY_OPERATORS)
    + tuple(_UNARY_OPERATORS)
    + tuple(_COMPARISON_OPERATORS)
)


def _unsupported(node):
    raise ExpressionSyntaxError(f"Unsupported syntax in Qlib expression: {type(node).__name__}")


def _get_operator(node):
    if not (
        isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "Operators"
    ):
        raise ExpressionSyntaxError("Only registered Qlib operators may be called")
    if node.func.attr.startswith("_"):
        raise ExpressionSyntaxError("Private operator names are not allowed")
    try:
        return Operators.get_operator(node.func.attr)
    except AttributeError as exc:
        raise ExpressionSyntaxError(f"Unknown Qlib operator: {node.func.attr}") from exc


def _validate(tree):
    # Check inactive branches too, without evaluating their arguments or operators.
    pending = [tree]
    while pending:
        node = pending.pop()
        if isinstance(node, ast.Call):
            _get_operator(node)
            names = [keyword.arg for keyword in node.keywords if keyword.arg is not None]
            if len(names) != len(set(names)):
                raise ExpressionSyntaxError("Repeated keyword argument")
            pending.extend(node.args)
            pending.extend(node.keywords)
        elif isinstance(node, _SUPPORTED_NODES):
            if (
                isinstance(node, ast.BoolOp)
                or isinstance(node, ast.UnaryOp)
                and isinstance(node.op, ast.Not)
                or isinstance(node, ast.Compare)
                and len(node.ops) > 1
            ):
                _require_literal(node)
            elif isinstance(node, ast.IfExp):
                _require_literal(node.test)
            pending.extend(ast.iter_child_nodes(node))
        else:
            _unsupported(node)


def _check_size(value):
    if len(value) > _MAX_CONTAINER_ITEMS:
        raise ExpressionSyntaxError(f"Containers and argument lists are limited to {_MAX_CONTAINER_ITEMS} items")
    return value


def _require_literal(node):
    if any(isinstance(child, ast.Call) for child in ast.walk(node)):
        raise ExpressionSyntaxError("Scalar conditions cannot contain Qlib operators; use If, & or | for expressions")


def _check_scalar(value):
    if type(value) not in _SCALAR_TYPES:
        raise ExpressionSyntaxError("This operation requires a literal scalar value")
    return value


def _scalar(node):
    return _check_scalar(_evaluate(node))


def _sequence(nodes):
    values = []
    for node in nodes:
        if isinstance(node, ast.Starred):
            items = _evaluate(node.value)
            if type(items) not in (list, tuple):
                raise ExpressionSyntaxError("Starred arguments require a literal list or tuple")
            values.extend(_check_size(items))
        else:
            values.append(_evaluate(node))
        _check_size(values)
    return values


def _mapping(pairs, keyword_args=False):
    values = {}
    for key_node, value_node in pairs:
        if key_node is None:
            update = _evaluate(value_node)
            if type(update) is not dict:
                raise ExpressionSyntaxError("Mapping expansion requires a literal dictionary")
            items = _check_size(update).items()
        else:
            key = _check_scalar(_evaluate(key_node))
            items = [(key, _evaluate(value_node))]
        for key, value in items:
            _check_scalar(key)
            if keyword_args:
                if type(key) is not str:
                    raise ExpressionSyntaxError("Keyword argument names must be strings")
                if key in values:
                    raise ExpressionSyntaxError(f"Repeated keyword argument: {key!r}")
            values[key] = value
            _check_size(values)
    return values


def _subscript(node):
    value = _evaluate(node.value)
    if type(value) not in (list, tuple, dict, str, bytes):
        raise ExpressionSyntaxError("Indexing is only supported on literal containers, not Qlib expressions")
    index = node.slice
    if isinstance(index, ast.Index):  # Python 3.8 wraps non-slice indices.
        index = index.value
    if isinstance(index, ast.Slice):
        bounds = [_evaluate(part) if part is not None else None for part in (index.lower, index.upper, index.step)]
        if type(value) is dict or any(part is not None and type(part) not in (int, bool) for part in bounds):
            raise ExpressionSyntaxError("Sequence slices require integer bounds")
        index = slice(*bounds)
    else:
        index = _check_scalar(_evaluate(index))
        if type(value) is not dict and type(index) not in (int, bool):
            raise ExpressionSyntaxError("Sequence indices must be integers")
    try:
        return value[index]
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        raise ExpressionSyntaxError(f"Invalid literal container index: {index!r}") from exc


def _constant_arithmetic(node, operation, left, right):
    """Evaluate numeric parameters without enabling string/list repetition."""
    for value in (left, right):
        if type(value) not in (int, float):
            raise ExpressionSyntaxError("Constant arithmetic requires real numbers")
        if isinstance(value, int) and value.bit_length() > 4096:
            raise ExpressionSyntaxError("Constant arithmetic exceeds the integer size limit")
    if isinstance(node.op, ast.Pow) and abs(right) > 4096:
        raise ExpressionSyntaxError("Constant exponent exceeds the size limit")
    if isinstance(node.op, ast.Pow) and isinstance(left, int) and isinstance(right, int) and right > 0:
        if left.bit_length() * right > 4096:
            raise ExpressionSyntaxError("Constant arithmetic exceeds the integer size limit")
    result = operation(left, right)
    if isinstance(result, int) and result.bit_length() > 4096:
        raise ExpressionSyntaxError("Constant arithmetic exceeds the integer size limit")
    if type(result) not in (int, float):
        raise ExpressionSyntaxError("Constant arithmetic must produce a real number")
    return result


def _evaluate(node):
    if isinstance(node, ast.Constant):
        return node.value

    if isinstance(node, ast.List):
        return _sequence(node.elts)

    if isinstance(node, ast.Tuple):
        return tuple(_sequence(node.elts))

    if isinstance(node, ast.Dict):
        return _mapping(zip(node.keys, node.values))

    if isinstance(node, ast.Subscript):
        return _subscript(node)

    if isinstance(node, ast.IfExp):
        return _evaluate(node.body if _scalar(node.test) else node.orelse)

    if isinstance(node, ast.BoolOp):
        value = _scalar(node.values[0])
        for item in node.values[1:]:
            if (isinstance(node.op, ast.And) and not value) or (isinstance(node.op, ast.Or) and value):
                break
            value = _scalar(item)
        return value

    if isinstance(node, ast.BinOp):
        operation = _BINARY_OPERATORS.get(type(node.op))
        if operation is None:
            _unsupported(node.op)
        left = _evaluate(node.left)
        right = _evaluate(node.right)
        if not isinstance(left, Expression) and not isinstance(right, Expression):
            return _constant_arithmetic(node, operation, left, right)
        return operation(left, right)

    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.Not):
            return not _scalar(node.operand)
        operation = _UNARY_OPERATORS.get(type(node.op))
        if operation is None:
            _unsupported(node.op)
        return operation(_evaluate(node.operand))

    if isinstance(node, ast.Compare):
        left = _evaluate(node.left)
        for op, comparator in zip(node.ops, node.comparators):
            operation = _COMPARISON_OPERATORS.get(type(op))
            if operation is None:
                _unsupported(op)
            right = _evaluate(comparator)
            if not isinstance(left, Expression) and not isinstance(right, Expression):
                _check_scalar(left)
                _check_scalar(right)
            result = operation(left, right)
            if len(node.ops) == 1:
                return result
            if not result:
                return False
            left = right
        return True

    if isinstance(node, ast.Call):
        operation = _get_operator(node)
        args = _sequence(node.args)
        kwargs = _mapping(
            (
                (ast.Constant(value=keyword.arg) if keyword.arg is not None else None, keyword.value)
                for keyword in node.keywords
            ),
            keyword_args=True,
        )
        _check_size(args + list(kwargs))
        return operation(*args, **kwargs)

    _unsupported(node)


def parse_expression(field) -> Expression:
    """Parse a Qlib expression without executing arbitrary Python code."""
    if isinstance(field, Expression):
        return field

    try:
        source = parse_field(field).strip()
        tree = ast.parse(source, mode="eval")
    except (SyntaxError, tokenize.TokenError) as exc:
        raise ExpressionSyntaxError(f"Invalid Qlib expression syntax: {field!r}") from exc

    _validate(tree)
    expression = _evaluate(tree.body)
    if not isinstance(expression, Expression):
        raise ExpressionSyntaxError("A Qlib expression must produce an Expression object")
    return expression
