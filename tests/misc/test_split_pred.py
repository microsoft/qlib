import pandas as pd
import pytest

from qlib.utils import split_pred


def _make_pred():
    dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    index = pd.MultiIndex.from_product([["SH600000", "SH600001"], dates], names=["instrument", "datetime"])
    return pd.DataFrame({"score": range(len(index))}, index=index)


def test_split_pred_with_all_dates():
    pred = _make_pred()

    pred_left, pred_right = split_pred(pred, number=3)

    assert pred_left.equals(pred.sort_index())
    assert pred_right.empty


def test_split_pred_rejects_non_positive_number():
    pred = _make_pred()

    with pytest.raises(ValueError, match="positive integer"):
        split_pred(pred, number=0)


def test_split_pred_rejects_number_larger_than_available_dates():
    pred = _make_pred()

    with pytest.raises(ValueError, match="greater than the number of available dates"):
        split_pred(pred, number=4)


def test_split_pred_rejects_number_larger_than_left_window():
    pred = _make_pred()

    with pytest.raises(ValueError, match="on or before `split_date`"):
        split_pred(pred, number=2, split_date="2020-01-01")
