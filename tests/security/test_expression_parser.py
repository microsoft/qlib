import pytest

from qlib.config import C
from qlib.data.base import Expression
from qlib.data.expression_parser import ExpressionSyntaxError, parse_expression
from qlib.data.ops import Operators, register_all_ops


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


@pytest.mark.parametrize("source", ["reset()", "register([])", "get_operator('Feature')", "If($close, reset(), $open)"])
def test_expression_cannot_call_operator_registry_methods(source, monkeypatch):
    from unittest.mock import Mock

    reset = Mock(wraps=Operators.reset)
    register = Mock(wraps=Operators.register)
    monkeypatch.setattr(Operators, "reset", reset)
    monkeypatch.setattr(Operators, "register", register)
    with pytest.raises(ExpressionSyntaxError, match="Unknown Qlib operator"):
        parse_expression(source)
    reset.assert_not_called()
    register.assert_not_called()
    assert isinstance(parse_expression("$close"), Expression)


@pytest.mark.parametrize(
    "source", ["Ref(*[$close, 1])", "Ref($close, **{})", "$close.__class__", "$close[0]", "unknown($close)"]
)
def test_expression_rejects_unsupported_calls_and_attributes(source):
    with pytest.raises(ExpressionSyntaxError):
        parse_expression(source)


def test_expression_provider_rejects_code_without_side_effects(tmp_path):
    from qlib.data.data import LocalExpressionProvider

    marker = tmp_path / "executed.txt"
    source = f"__import__('pathlib').Path({str(marker)!r}).touch()"
    provider = LocalExpressionProvider()
    with pytest.raises(ExpressionSyntaxError):
        provider.get_expression_instance(source)
    assert not marker.exists()


def test_expression_computes_expected_values(monkeypatch):
    from types import SimpleNamespace
    import pandas as pd
    from qlib.data import data

    close = pd.Series([2.0, 4.0, 8.0], index=[0, 1, 2], dtype="float32")
    monkeypatch.setattr(data, "FeatureD", SimpleNamespace(feature=lambda *args, **kwargs: close))
    actual = parse_expression("Ref($close, 1) / $close - 1").load("TEST", 0, 2, "day")
    expected = pd.Series([float("nan"), -0.5, -0.5], index=[0, 1, 2], dtype="float32")
    pd.testing.assert_series_equal(actual, expected, check_names=False)


def test_registered_custom_operator_is_supported():
    from qlib.data.ops import Ref

    class CustomRef(Ref):
        pass

    Operators.register([CustomRef])
    assert isinstance(parse_expression("CustomRef($close, 1)"), CustomRef)


@pytest.mark.parametrize(
    "source, equivalent",
    [
        ("Mean($close, 2 + 3)", "Mean($close, 5)"),
        ("Ref($close, 2 * 3)", "Ref($close, 6)"),
        ("$close / (1 + 0.01)", "$close / 1.01"),
        ("$close + 10 ** -6", "$close + 0.000001"),
        ("Ref($close, (60 // 5) % 5)", "Ref($close, 2)"),
    ],
)
def test_numeric_parameter_arithmetic_preserves_expression(source, equivalent):
    assert str(parse_expression(source)) == str(parse_expression(equivalent))


@pytest.mark.parametrize(
    "source",
    [
        "Ref($close, 'x' * 1000000000)",
        "Ref($close, [1] * 1000000000)",
        "Ref($close, 2 ** 1000000000)",
        "Ref($close, (2 ** 4095) * (2 ** 4095))",
        "$close + (-1) ** 0.5",
    ],
)
def test_constant_arithmetic_rejects_expansion_and_oversized_results(source):
    with pytest.raises(ExpressionSyntaxError):
        parse_expression(source)


def test_benchmark_feature_expressions_remain_supported():
    from qlib.contrib.data.loader import Alpha158DL, Alpha360DL

    for loader in (Alpha158DL, Alpha360DL):
        fields, _ = loader.get_feature_config()
        for field in fields:
            assert isinstance(parse_expression(field), Expression), field


def test_disk_cache_rejects_code_before_querying_provider(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from qlib.data import data
    from qlib.data.cache import DiskExpressionCache

    marker = tmp_path / "executed.txt"
    source = f"__import__('pathlib').Path({str(marker)!r}).touch()"
    cache = object.__new__(DiskExpressionCache)
    cache.provider = Mock()
    monkeypatch.setattr(cache, "get_cache_dir", lambda freq: tmp_path / "cache")
    monkeypatch.setattr(
        data,
        "Cal",
        SimpleNamespace(
            calendar=lambda **kwargs: [0, 1],
            locate_index=lambda *args, **kwargs: (0, 1, 0, 1),
        ),
    )
    with pytest.raises(ExpressionSyntaxError):
        cache._expression("TEST", source)
    cache.provider.expression.assert_not_called()
    assert not marker.exists()
