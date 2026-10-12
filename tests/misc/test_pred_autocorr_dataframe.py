import pandas as pd

from qlib.contrib.eva.alpha import pred_autocorr


def test_pred_autocorr_dataframe_uses_first_column():
    index = pd.MultiIndex.from_product(
        [list("ABC"), pd.date_range("2024-01-02", periods=3, freq="B")],
        names=["instrument", "datetime"],
    )
    first = pd.Series([1, 2, 3, 2, 3, 1, 3, 1, 2], index=index, name="score")
    predictions = pd.DataFrame({"score": first, "unused": -first})

    pd.testing.assert_series_equal(pred_autocorr(predictions), pred_autocorr(first))
