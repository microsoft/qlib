import pandas as pd
import pytest

from scripts.data_collector.yahoo.collector import YahooNormalize


def test_calc_change_uses_adjusted_close_across_split() -> None:
    frame = pd.DataFrame(
        {
            "close": [100.0, 50.0],
            "adjclose": [25.0, 25.0],
        }
    )

    change = YahooNormalize.calc_change(frame, last_close=None)

    assert pd.isna(change.iloc[0])
    assert change.iloc[1] == pytest.approx(0.0)


def test_calc_change_falls_back_to_close() -> None:
    frame = pd.DataFrame({"close": [100.0, 110.0]})

    change = YahooNormalize.calc_change(frame, last_close=None)

    assert pd.isna(change.iloc[0])
    assert change.iloc[1] == pytest.approx(0.1)
