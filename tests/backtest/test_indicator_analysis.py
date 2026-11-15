# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pandas as pd
import pytest

from qlib.contrib.evaluate import indicator_analysis


@pytest.fixture
def indicators():
    return pd.DataFrame(
        {
            "ffr": [0.5, 1.0],
            "pa": [0.02, -0.01],
            "pos": [1.0, 0.0],
            "count": [1, 3],
            "deal_amount": [-2.0, 1.0],
            "value": [1.0, -4.0],
        }
    )


@pytest.mark.parametrize(
    "method,optional_columns,expected_ffr,expected_pa",
    [
        ("mean", [], 0.875, -0.0025),
        ("amount_weighted", ["deal_amount"], 2.0 / 3.0, 0.01),
        ("value_weighted", ["value"], 0.9, -0.004),
    ],
)
@pytest.mark.parametrize("minimal", [False, True])
def test_indicator_analysis_weights(indicators, method, optional_columns, expected_ffr, expected_pa, minimal):
    if minimal:
        indicators = indicators[["ffr", "pa", "pos", "count"] + optional_columns]
    expected = pd.DataFrame({"value": [expected_ffr, expected_pa, 0.25]}, index=["ffr", "pa", "pos"])
    pd.testing.assert_frame_equal(indicator_analysis(indicators, method), expected)


def test_indicator_analysis_default_method(indicators):
    minimal = indicators[["ffr", "pa", "pos", "count"]]
    pd.testing.assert_frame_equal(indicator_analysis(minimal), indicator_analysis(indicators, "mean"))


@pytest.mark.parametrize("columns", [[], ["ffr", "pa", "pos", "count"], None])
def test_indicator_analysis_invalid_method(indicators, columns):
    if columns is not None:
        indicators = indicators[columns]
    with pytest.raises(ValueError, match="indicator_analysis method unsupported is not supported"):
        indicator_analysis(indicators, "unsupported")


@pytest.mark.parametrize("method,column", [("amount_weighted", "deal_amount"), ("value_weighted", "value")])
def test_indicator_analysis_requires_selected_weight(indicators, method, column):
    with pytest.raises(KeyError, match=column):
        indicator_analysis(indicators.drop(columns=column), method)


@pytest.mark.parametrize("method", ["mean", "amount_weighted", "value_weighted"])
def test_indicator_analysis_requires_count_for_pos(indicators, method):
    with pytest.raises(KeyError, match="count"):
        indicator_analysis(indicators.drop(columns="count"), method)


@pytest.mark.parametrize("method", ["mean", "amount_weighted", "value_weighted"])
def test_indicator_analysis_empty_input(indicators, method):
    result = indicator_analysis(indicators.iloc[:0], method)
    assert result.index.tolist() == ["ffr", "pa", "pos"]
    assert result.columns.tolist() == ["value"]
    assert result.isna().all().all()


@pytest.mark.parametrize(
    "method,unused_columns",
    [("mean", ["deal_amount", "value"]), ("amount_weighted", ["value"]), ("value_weighted", ["deal_amount"])],
)
def test_indicator_analysis_ignores_unused_weights(indicators, method, unused_columns):
    expected = indicator_analysis(indicators, method)
    indicators[unused_columns] = "unused"
    pd.testing.assert_frame_equal(indicator_analysis(indicators, method), expected)
