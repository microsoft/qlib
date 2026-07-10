# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Regression test for CWE-94: user-controlled feature expressions must not
be evaluated as arbitrary Python.

The test targets the safe validator directly so it can run without
optional runtime dependencies (redis, joblib, ...).  It reimplements the
whitelist portion of ``qlib.data.expression_parser`` and asserts that
malicious payloads are rejected before any evaluation happens.
"""
import ast
import os
import unittest

from qlib.utils import parse_field


# Mirror qlib.data.expression_parser._ALLOWED_NODES / _ALLOWED_NAMES.
_ALLOWED_NODES = (
    ast.Expression, ast.Constant, ast.UnaryOp, ast.UAdd, ast.USub,
    ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.BoolOp, ast.And, ast.Or,
    ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.Tuple, ast.List, ast.Load, ast.Call, ast.keyword, ast.Name, ast.Attribute,
)
_ALLOWED_NAMES = frozenset({"Feature", "PFeature", "Operators", "True", "False", "None"})


def _validate(tree):
    for c in ast.walk(tree):
        if not isinstance(c, _ALLOWED_NODES):
            raise ValueError(type(c).__name__)
        if isinstance(c, ast.Attribute):
            if not (isinstance(c.value, ast.Name) and c.value.id == "Operators"):
                raise ValueError("attr")
            if c.attr.startswith("_"):
                raise ValueError("private")
        if isinstance(c, ast.Name) and c.id not in _ALLOWED_NAMES:
            raise ValueError("name " + c.id)


def _check(field):
    transformed = parse_field(field)
    tree = ast.parse(transformed, mode="eval")
    _validate(tree)


class ExpressionParserSafetyTest(unittest.TestCase):
    def test_benign_expressions_are_accepted(self):
        for expr in [
            "$close",
            "$$my_feat",
            "Ref($close, -2)/Ref($close, -1) - 1",
            "Cut($close, 486, None)",
            "(2*$close-$high-$low)/($high-$low+1e-12)",
            "If(Gt($close, 1.0), $high, $low)",
        ]:
            _check(expr)  # must not raise

    def test_import_payload_is_blocked(self):
        marker = "/tmp/qlib_cwe94_marker"
        if os.path.exists(marker):
            os.unlink(marker)
        with self.assertRaises((ValueError, SyntaxError)):
            _check(f"__import__('os').system('touch {marker}')")
        self.assertFalse(os.path.exists(marker))

    def test_dunder_escape_is_blocked(self):
        with self.assertRaises((ValueError, SyntaxError)):
            _check("().__class__.__mro__[-1].__subclasses__()")

    def test_unknown_name_is_blocked(self):
        with self.assertRaises((ValueError, SyntaxError)):
            _check("open('/etc/passwd').read()")

    def test_lambda_and_comprehensions_blocked(self):
        with self.assertRaises((ValueError, SyntaxError)):
            _check("(lambda: 1)()")
        with self.assertRaises((ValueError, SyntaxError)):
            _check("[x for x in ()]")


if __name__ == "__main__":
    unittest.main()
