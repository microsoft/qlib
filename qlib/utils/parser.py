# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import re
from qlib.data.base import Expression

# Functions from qlib.data.base and qlib.data.ops that parse_field implicitly prepares for (e.g., Feature, PFeature, Operators)
# are not directly imported here as parse_field only performs string manipulation.
# The eval context (e.g., in LocalExpressionProvider) is responsible for having those names in scope.

def parse_field(field: str) -> str:
    """
    Parse field string to qlib expression.
    Field names are expected to consist of Latin letters, numbers, and underscores.

    Parameters
    ----------
    field : str
        field string

    Returns
    -------
    str
        qlib expression
    """
    if not isinstance(field, str):
        field = str(field)

    # Order matters: $$ must be checked before $
    # Operators should be checked last as their names might be substrings of feature names if not careful,
    # though the '(' helps distinguish.
    patterns = [
        (
            r"\$\$([\w]+)",  # Match $$ followed by one or more word characters
            r'PFeature("\1")',
        ),
        (
            r"\$([\w]+)",   # Match $ followed by one or more word characters
            r'Feature("\1")'
        ),
        (
            r"([a-zA-Z_][\w]*\s*)\((", # Match OperatorName( - captures OperatorName, handles optional space
            r"Operators.\1(",
        ),
    ]

    for pattern, new in patterns:
        field = re.sub(pattern, new, field)
    return field

def analyze_expression_raw_features(expression_object: Expression) -> set:
    """
    Analyzes a given Expression object and returns a set of raw feature names it depends on.

    Parameters
    ----------
    expression_object : Expression
        The instantiated qlib Expression object.

    Returns
    -------
    set
        A set of strings, where each string is a raw feature name (e.g., '$close', '$$roe_q').
    """
    if not isinstance(expression_object, Expression):
        raise TypeError("Input must be a qlib.data.base.Expression object.")
    return expression_object.get_required_raw_features() 