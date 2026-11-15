# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import pytest

from qlib.backtest.high_performance_ds import NumpyQuote, PandasQuote


def make_quotes(values, reverse=False):
    dates = pd.date_range("2024-01-02", periods=3, name="datetime")
    index = pd.MultiIndex.from_product([["stock"], dates], names=["instrument", "datetime"])
    frame = pd.DataFrame({"$close": values}, index=index)
    if reverse:
        frame = frame.iloc[::-1]
    return PandasQuote(frame, "day"), NumpyQuote(frame, "day")


@pytest.mark.parametrize("values", [[1.0, 2.0, 3.0], [1.0, np.nan, 3.0], [1.0, 2.0, np.nan], [np.nan] * 3])
@pytest.mark.parametrize("reverse", [False, True])
def test_pandas_quote_last_matches_numpy(values, reverse):
    pandas_quote, numpy_quote = make_quotes(values, reverse)
    args = ("stock", pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-04"), "$close")
    actual = pandas_quote.get_data(*args, method="last")
    expected = values[-1]
    if np.isnan(expected):
        assert np.isnan(actual)
        assert np.isnan(numpy_quote.get_data(*args, method="last"))
    else:
        assert actual == expected == numpy_quote.get_data(*args, method="last")


@pytest.mark.parametrize(
    "start,end,expected",
    [
        ("2024-01-02", "2024-01-03", 2.0),
        ("2024-01-03", "2024-01-03", 2.0),
        (None, "2024-01-03", 2.0),
        ("2024-01-03", None, 3.0),
    ],
)
def test_pandas_quote_last_respects_time_window(start, end, expected):
    quote, _ = make_quotes([1.0, 2.0, 3.0])
    assert quote.get_data("stock", start, end, "$close", method="last") == expected


def test_pandas_quote_last_empty_window():
    quote, _ = make_quotes([1.0, 2.0, 3.0])
    assert quote.get_data("stock", "2024-02-01", "2024-02-02", "$close", method="last") is None


@pytest.mark.parametrize("method,expected", [("mean", 1.5), ("sum", 3.0), ("ts_data_last", 2.0)])
def test_pandas_quote_other_aggregations_unchanged(method, expected):
    quote, _ = make_quotes([1.0, 2.0, np.nan])
    assert quote.get_data("stock", "2024-01-02", "2024-01-04", "$close", method=method) == expected


def test_pandas_quote_no_aggregation_preserves_series():
    quote, _ = make_quotes([1.0, 2.0, 3.0])
    result = quote.get_data("stock", "2024-01-02", "2024-01-03", "$close", method=None)
    np.testing.assert_array_equal(result.data, [1.0, 2.0])
