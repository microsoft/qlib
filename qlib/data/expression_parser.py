# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Safe parser for qlib feature expression strings.

The qlib feature DSL lets users write expressions such as::

    Ref($close, -2) / Ref($close, -1) - 1

which are transformed by :func:`qlib.utils.parse_field` into::

    Operators.Ref(Feature("close"), -2) / Operators.Ref(Feature("close"), -1) - 1

Historically qlib evaluated the transformed string with the builtin
``eval`` which allowed arbitrary Python — an attacker who could influence
a field string (for example through a workflow YAML config) could gain
code execution (CWE-94).

This module implements a whitelist-based AST evaluator that only permits
the constructs actually used by the DSL: numeric/string/boolean/None
literals, tuples/lists, unary +/-, arithmetic and comparison operators,
and calls to a small set of allowed names (``Feature``, ``PFeature`` and
``Operators.<op>``).  Anything else — attribute traversal, arbitrary
name lookups, ``__import__``, generator/lambda/subscript-escape tricks —
raises :class:`ValueError`.
"""

from __future__ import annotations

import ast
from typing import Any, Dict

from ..utils import parse_field
from .base import Feature, PFeature
from .ops import Operators


# AST nodes that are structurally safe (contain no executable side effects
# on their own; their children are validated recursively).
_ALLOWED_NODES = (
    ast.Expression,
    ast.Constant,
    ast.Num,          # legacy alias (<3.8)
    ast.Str,          # legacy alias (<3.8)
    ast.NameConstant, # legacy alias (<3.8)
    ast.UnaryOp,
    ast.UAdd,
    ast.USub,
    ast.BinOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.Compare,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.Tuple,
    ast.List,
    ast.Load,
    ast.Call,
    ast.keyword,
    ast.Name,
    ast.Attribute,
)

# Only these bare names may appear in an expression.
_ALLOWED_NAMES = frozenset({"Feature", "PFeature", "Operators", "True", "False", "None"})


def _validate(node: ast.AST) -> None:
    for child in ast.walk(node):
        if not isinstance(child, _ALLOWED_NODES):
            raise ValueError(
                f"Disallowed expression element: {type(child).__name__}"
            )
        if isinstance(child, ast.Attribute):
            # Only ``Operators.<name>`` is allowed; the value must be the
            # bare name ``Operators``.
            if not (isinstance(child.value, ast.Name) and child.value.id == "Operators"):
                raise ValueError("Only attribute access on 'Operators' is allowed")
            if child.attr.startswith("_"):
                raise ValueError("Private attribute access is not allowed")
        if isinstance(child, ast.Name) and child.id not in _ALLOWED_NAMES:
            raise ValueError(f"Unknown name in expression: {child.id!r}")
        if isinstance(child, ast.Call):
            # keyword arguments are OK, but reject **kwargs / *args unpacking
            # that could smuggle arbitrary objects.
            for kw in child.keywords:
                if kw.arg is None:
                    raise ValueError("**kwargs is not allowed in expressions")


def _safe_namespace() -> Dict[str, Any]:
    return {
        "__builtins__": {},
        "Feature": Feature,
        "PFeature": PFeature,
        "Operators": Operators,
        "True": True,
        "False": False,
        "None": None,
    }


def parse_expression(field: str) -> Any:
    """Parse a qlib feature-DSL string and return the expression object.

    This is the safe replacement for ``eval(parse_field(field))``.

    Parameters
    ----------
    field:
        The user-facing feature expression, e.g. ``"Ref($close, -1)"``.

    Raises
    ------
    ValueError
        If ``field`` contains anything outside of the feature DSL.
    SyntaxError
        If the transformed expression is not valid Python.
    """
    transformed = parse_field(field)
    tree = ast.parse(transformed, mode="eval")
    _validate(tree)
    code = compile(tree, filename="<qlib-expression>", mode="eval")
    # pylint: disable=eval-used  # sandboxed: AST whitelist + no builtins
    return eval(code, _safe_namespace())  # noqa: S307
