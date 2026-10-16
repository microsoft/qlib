import logging

import pandas as pd

from qlib.contrib.eva.alpha import pred_autocorr


def test_pred_autocorr_dataframe_uses_first_column(caplog, monkeypatch):
    index = pd.MultiIndex.from_product(
        [list("ABC"), pd.date_range("2024-01-02", periods=3, freq="B")],
        names=["instrument", "datetime"],
    )
    first = pd.Series([1, 2, 3, 2, 3, 1, 3, 1, 2], index=index, name="score")
    # The second column must have different autocorrelations to detect selecting it.
    second = pd.Series([1, 2, 4, 1, 5, 2, 3, 1, 9], index=index)
    predictions = pd.DataFrame({"score": first, "unused": second})

    monkeypatch.setattr(logging.getLogger("qlib"), "propagate", True)
    pd.testing.assert_series_equal(pred_autocorr(predictions), pred_autocorr(first))
    assert (
        "qlib.pred_autocorr",
        logging.WARNING,
        f"Only the first column in {predictions.columns} of `pred` is kept",
    ) in caplog.record_tuples
