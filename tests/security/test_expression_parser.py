import pytest

from qlib.config import C
from qlib.data.base import Expression
from qlib.data.expression_parser import ExpressionSyntaxError, parse_expression
from qlib.data.ops import Operators, register_all_ops
from qlib.utils import remove_fields_space


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
        "P(Mean($$roewa_q, 2))",
        "ChangeInstrument('SH000300', $close)",
        "TResample($ask1, '1min', 'last')",
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
    "source", ["Ref(*$close)", "Ref($close, **$open)", "$close.__class__", "$close[0]", "unknown($close)"]
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


@pytest.mark.parametrize(
    "source",
    [
        "Ref($close, 1) / $close - 1",
        "Ref(*[$close], **{'N': [1, 2][0] if (1 < 2 and not False) else 2}) / $close - 1",
    ],
)
def test_expression_computes_expected_values(monkeypatch, source):
    from types import SimpleNamespace
    import pandas as pd
    from qlib.data import data

    close = pd.Series([2.0, 4.0, 8.0], index=[0, 1, 2], dtype="float32")
    monkeypatch.setattr(data, "FeatureD", SimpleNamespace(feature=lambda *args, **kwargs: close))
    actual = (
        data.LocalExpressionProvider().get_expression_instance(remove_fields_space(source)).load("TEST", 0, 2, "day")
    )
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
    "source, equivalent",
    [
        ("Ref(*[$close, 1])", "Ref($close, 1)"),
        ("Ref(*($close, 1))", "Ref($close, 1)"),
        ("Ref(*[$close, *[1]])", "Ref($close, 1)"),
        ("Ref(*($close, *(1,)))", "Ref($close, 1)"),
        ("Ref($close, **{'N': 1})", "Ref($close, 1)"),
        ("Ref(**{'feature': $close, 'N': 1})", "Ref($close, 1)"),
        ("Ref(*[$close], **{'N': 1})", "Ref($close, 1)"),
        ("Ref($close, **{'N': 2, **{'N': 1}})", "Ref($close, 1)"),
        ("Ref($close, [1, 2][0])", "Ref($close, 1)"),
        ("Ref($close, (1, 2)[-2])", "Ref($close, 1)"),
        ("Ref($close, {'window': 1}['window'])", "Ref($close, 1)"),
        ("Ref($close, {None: 1}[None])", "Ref($close, 1)"),
        ("Ref($close, {1.5: 1}[1.5])", "Ref($close, 1)"),
        ("Ref($close, {1: 2, True: 1}[1])", "Ref($close, 1)"),
        ("Ref(*[$close, 1, 99][:2])", "Ref($close, 1)"),
        ("Ref(*[1, $close][::-1])", "Ref($close, 1)"),
        ("Ref($close, [1, 2][None:None:None][0])", "Ref($close, 1)"),
        ("Feature('close'[::1])", "$close"),
        ("Ref($close, 1 if True else 2)", "Ref($close, 1)"),
        ("Ref($close, 1 if (2 > 1) else 2)", "Ref($close, 1)"),
        ("Ref($close, 1 if (1 < 2 and (2 < 3)) else 2)", "Ref($close, 1)"),
        ("Ref($close, 1 if not(False) else 2)", "Ref($close, 1)"),
        ("Ref($close, 0 or (1))", "Ref($close, 1)"),
        ("Ref($close, 1 and (2))", "Ref($close, 2)"),
        ("Ref($close, 1 if 0 < 1 < 2 else 2)", "Ref($close, 1)"),
        ("Ref($close, 1 if 'a' == 'a' else 2)", "Ref($close, 1)"),
        ("If(1 < 2, $close, $open)", "If(True, $close, $open)"),
        ("$close if True else $open", "$close"),
        ("Ref($close, 1 if True else 1 / 0)", "Ref($close, 1)"),
        ("Ref($close, 0 and (1 / 0))", "Ref($close, 0)"),
        ("Ref($close, 1 if 3 < 2 < (1 / 0) else 2)", "Ref($close, 2)"),
        ("  Ref( $close, 1 )  ", "Ref($close, 1)"),
        ("\tRef($close, 1)\t", "Ref($close, 1)"),
    ],
)
def test_safe_python_conveniences_preserve_expression(source, equivalent):
    assert str(parse_expression(source)) == str(parse_expression(equivalent))
    assert str(parse_expression(remove_fields_space(source))) == str(parse_expression(equivalent))


@pytest.fixture
def capture_operator():
    from qlib.data.ops import Ref

    class Capture(Ref):
        def __init__(self, *args, **kwargs):
            super().__init__(args[0], 1)
            self.args = args
            self.kwargs = kwargs

    Operators.register([Capture])
    return Capture


@pytest.mark.parametrize(
    "value",
    [
        "$close",
        "Mean($close, 5)",
        "if (x) and (y)",
        "a'b\"c",
        "line1\nline2",
        "\u4e2d\u6587\uff08\u5b57\u6bb5\uff09",
        b"$close",
    ],
)
def test_string_arguments_are_not_rewritten(capture_operator, value):
    source = f"Capture($close, {value!r})"
    for field in (source, remove_fields_space(source)):
        expression = parse_expression(field)
        assert isinstance(expression, capture_operator)
        assert expression.args[1] == value


def test_nested_literal_parameters(capture_operator):
    expression = parse_expression(
        "Capture($close, {'weights': [0.2, 0.8], 'options': (None, True), 'feature': $open}, "
        "**{'window': 5, 'label': '$close'})"
    )
    assert expression.args[1]["weights"] == [0.2, 0.8]
    assert expression.args[1]["options"] == (None, True)
    assert str(expression.args[1]["feature"]) == "$open"
    assert expression.kwargs == {"window": 5, "label": "$close"}


def test_multiline_and_adjacent_string_literals(capture_operator):
    expression = parse_expression("Capture(\n$close,\n'''literal\n$open(foo)''',\n'c' 'lose'\n)")
    assert expression.args[1:] == ("literal\n$open(foo)", "close")


@pytest.mark.parametrize(
    "source",
    [
        "Ref(*'close')",
        "Ref(*{'feature': $close})",
        "Ref($close, **[])",
        "Ref($close, **{1: 1})",
        "Ref($close, N=1, N=2)",
        "Ref($close, N=1, **{'N': 2})",
        "Ref($close, **{'N': 1}, **{'N': 2})",
        "Ref($close, [1][3])",
        "Ref($close, [1]['0'])",
        "Ref($close, [1][::0])",
        "Ref($close, {'window': 1}['missing'])",
        "Ref($close, {$close: 1}['window'])",
        "Ref($close, {(1, 2): 1}[(1, 2)])",
        "Ref($close, [1][$close])",
        "Ref($close, [1][:$close])",
        "Ref($close, 1 if $close else 2)",
        "Ref($close, 1 if False and $close else 2)",
        "$close and $open",
        "$close or $open",
        "not $close",
        "0 < $close < 10",
        "$close if $open > 0 else $high",
        "Ref($close, [1] == [2])",
        "Ref($close, [item for item in [1]])",
        "Ref($close, 1 if True else reset())",
        "Ref($close, True or reset())",
        "Ref($close, 1 if True else $open.__class__)",
        "Ref($close, 1 if True else (lambda: 2))",
        "Ref($close, f'{1}')",
        "Ref($close, (x := 1))",
        "$close if True else ($open and $high)",
        "$close if True else (0 < $open < 10)",
        "$close if True else ($open if $volume else $high)",
        "$close if True else Ref($open, N=1, N=2)",
    ],
)
def test_compatibility_does_not_expose_python_objects(source):
    with pytest.raises(ExpressionSyntaxError):
        parse_expression(source)


def test_expression_protocols_are_not_used_for_literal_operations(monkeypatch):
    from unittest.mock import Mock
    from qlib.data.base import Feature

    hooks = []
    for name in ("__bool__", "__iter__", "__getitem__", "__hash__"):
        hook = Mock(side_effect=AssertionError(f"{name} must not run"))
        monkeypatch.setattr(Feature, name, hook, raising=False)
        hooks.append(hook)
    for source in ("Ref(*$close)", "$close[0]", "Ref($close, {$close: 1})", "$close if $open else $high"):
        with pytest.raises(ExpressionSyntaxError):
            parse_expression(source)
    for hook in hooks:
        hook.assert_not_called()


@pytest.mark.parametrize("source", ["Ref($close, **ForeignMapping())", "Ref($close, ForeignMapping()['N'])"])
def test_literal_mapping_operations_reject_subclass_protocols(source):
    from qlib.data.ops import Ref

    class ForeignDict(dict):
        def items(self):
            raise AssertionError("custom mapping protocol must not run")

        def __getitem__(self, key):
            raise AssertionError("custom indexing protocol must not run")

    class ForeignMapping(Ref):
        def __new__(cls):
            return ForeignDict(N=1)

    Operators.register([ForeignMapping])
    with pytest.raises(ExpressionSyntaxError):
        parse_expression(source)


def test_inactive_branches_are_validated_without_calling_operators():
    from unittest.mock import Mock
    from qlib.data.ops import Ref

    constructed = Mock()

    class TrackingRef(Ref):
        def __init__(self, *args):
            constructed()
            super().__init__(*args)

    Operators.register([TrackingRef])
    assert str(parse_expression("$close if True else TrackingRef($open, 1)")) == "$close"
    with pytest.raises(ExpressionSyntaxError, match="Unknown Qlib operator"):
        parse_expression("TrackingRef($close, 1 if True else unknown($open))")
    constructed.assert_not_called()


@pytest.mark.parametrize("kind", ["list", "tuple", "dict", "args", "kwargs"])
@pytest.mark.parametrize("extra", [0, 1])
def test_container_and_argument_size_boundaries(capture_operator, kind, extra):
    from qlib.data.expression_parser import _MAX_CONTAINER_ITEMS

    size = _MAX_CONTAINER_ITEMS + extra
    if kind in ("args", "kwargs"):
        size -= 1  # The feature itself is one argument.
    sequence = ",".join("0" for _ in range(size))
    mapping = ",".join(f"'k{i}':0" for i in range(size))
    sources = {
        "list": f"Capture($close, [{sequence}])",
        "tuple": f"Capture($close, ({sequence},))",
        "dict": f"Capture($close, {{{mapping}}})",
        "args": f"Capture($close, *[{sequence}])",
        "kwargs": f"Capture($close, **{{{mapping}}})",
    }
    if extra:
        with pytest.raises(ExpressionSyntaxError, match="limited to"):
            parse_expression(sources[kind])
    else:
        expression = parse_expression(sources[kind])
        if kind == "args":
            assert len(expression.args) == _MAX_CONTAINER_ITEMS
        elif kind == "kwargs":
            assert len(expression.args) + len(expression.kwargs) == _MAX_CONTAINER_ITEMS
        else:
            assert len(expression.args[1]) == _MAX_CONTAINER_ITEMS


def test_expanded_containers_obey_size_limit(capture_operator, monkeypatch):
    monkeypatch.setattr("qlib.data.expression_parser._MAX_CONTAINER_ITEMS", 2)
    for source in ("Capture($close, [*[1, 2], 3])", "Capture($close, {**{'a': 1, 'b': 2}, 'c': 3})"):
        with pytest.raises(ExpressionSyntaxError, match="limited to"):
            parse_expression(source)


@pytest.mark.parametrize("source", ["Ref($close,", "Ref($close, 'unfinished)", "Ref($close, 1]"])
def test_malformed_tokens_raise_expression_syntax_error(source):
    with pytest.raises(ExpressionSyntaxError, match="Invalid Qlib expression syntax"):
        parse_expression(source)


def test_expression_error_links_migration_after_exception_roundtrip():
    import pickle
    from qlib.utils.mod import CONFIG_MIGRATION_GUIDE

    with pytest.raises(ExpressionSyntaxError) as error:
        parse_expression("$close.__class__")
    restored = pickle.loads(pickle.dumps(error.value))
    assert str(restored) == str(error.value)
    assert str(restored).count(CONFIG_MIGRATION_GUIDE) == 1


@pytest.mark.parametrize("source", ["Ref($close, 2 * * 3)", "Ref($close, 1 . 2)", "Ref($close, r 'x')"])
def test_normalization_does_not_merge_distinct_tokens(source):
    with pytest.raises(ExpressionSyntaxError):
        parse_expression(remove_fields_space(source))


def test_normalization_preserves_container_and_literal_semantics(capture_operator):
    from qlib.utils import normalize_cache_fields

    sources = ["Capture($close, {'a b': 1}['a b'])", "Capture($close, 'a b')", "Capture($close, 'ab')"]
    normalized = normalize_cache_fields(sources)
    assert len(normalized) == 3
    for source in sources:
        assert parse_expression(remove_fields_space(source)).args[1] == parse_expression(source).args[1]
    assert remove_fields_space(("Ref($close, 1)",)) == ["Ref($close,1)"]
    assert remove_fields_space("  Ref( $close, 1 )  ") == "Ref($close,1)"


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
    from qlib.utils import parse_field

    for loader in (Alpha158DL, Alpha360DL):
        fields, _ = loader.get_feature_config()
        for field in fields:
            parsed = parse_expression(field)
            # Only the fixed, trusted benchmark definitions are evaluated as Python.
            original = eval(parse_field(field), {"Operators": Operators, "__builtins__": {}})
            assert isinstance(parsed, Expression), field
            assert str(parsed) == str(original), field
            assert parsed.get_extended_window_size() == original.get_extended_window_size(), field
            assert remove_fields_space(field) == field.replace(" ", ""), field


def test_disk_cache_accepts_conditional_feature_names(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from qlib.data import data
    from qlib.data.cache import DiskExpressionCache

    source = "Feature('close price' if (1 < 2 and not False) else 'open price')"
    cache = object.__new__(DiskExpressionCache)
    cache.provider = Mock()
    monkeypatch.setattr(cache, "get_cache_dir", lambda freq: tmp_path / "cache")
    monkeypatch.setattr(
        data,
        "Cal",
        SimpleNamespace(calendar=lambda **kwargs: [0, 1], locate_index=lambda *args, **kwargs: (0, 1, 0, 1)),
    )
    assert cache._expression("TEST", source) is cache.provider.expression.return_value
    cache.provider.expression.assert_called_once_with("TEST", remove_fields_space(source), None, None, "day")
    assert cache._uri("TEST", "Feature('close price')", None, None, "day") != cache._uri(
        "TEST", "Feature('closeprice')", None, None, "day"
    )


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
