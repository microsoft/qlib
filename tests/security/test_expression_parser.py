import pytest

from qlib.config import C
from qlib.data.base import Expression
from qlib.data.expression_parser import ExpressionSyntaxError, parse_expression
from qlib.data.ops import register_all_ops


@pytest.fixture(autouse=True)
def register_operators():
    register_all_ops(C)


@pytest.mark.parametrize(
    "source",
    [
        "$close",
        "Ref($close, -1)",
        "Ref($close, 1) / $close - 1",
        "Mean($close, 5)",
        "If(Gt($close, $open), $close, $open)",
        "($close > $open) & ($volume > 0)",
    ],
)
def test_parse_expression_supports_qlib_syntax(source):
    assert isinstance(parse_expression(source), Expression)


@pytest.mark.parametrize(
    "source",
    [
        '__import__("os").system("id")',
        "(lambda: 0).__globals__",
        "[].__class__.__mro__",
        "[item for item in range(10)]",
        'getattr(Operators, "Feature")("close")',
        'Operators.__getattribute__("Feature")',
    ],
)
def test_parse_expression_rejects_python_execution_syntax(source):
    with pytest.raises(ExpressionSyntaxError):
        parse_expression(source)
