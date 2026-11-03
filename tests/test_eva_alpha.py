"""Tests for qlib.contrib.eva.alpha."""

import pandas as pd

from qlib.contrib.eva.alpha import pred_autocorr


def _sample_pred() -> pd.Series:
    index = pd.MultiIndex.from_product(
        [list("ABC"), pd.date_range("2024-01-02", periods=3, freq="B")],
        names=["instrument", "datetime"],
    )
    return pd.Series([1, 2, 3, 2, 3, 1, 3, 1, 2], index=index)


def test_pred_autocorr_accepts_dataframe():
    """DataFrame input should keep the first column instead of raising AttributeError."""
    score = _sample_pred()
    frame = pd.DataFrame({"score": score, "unused": -score})

    result = pred_autocorr(frame)

    pd.testing.assert_series_equal(result, pred_autocorr(score))
