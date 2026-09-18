import pandas as pd
import pytest

from scripts.data_collector.yahoo.collector import YahooNormalize


def test_normalize_yahoo_accepts_mixed_daily_and_timezone_dates():
    df = pd.DataFrame(
        {
            "date": ["2025-09-01", "2025-09-02 09:30:00+08:00"],
            "symbol": ["sh600468", "sh600468"],
            "open": [10.0, 11.0],
            "close": [10.0, 11.0],
            "high": [10.0, 11.0],
            "low": [10.0, 11.0],
            "volume": [100.0, 100.0],
        }
    )

    result = YahooNormalize.normalize_yahoo(df)

    assert result["date"].tolist() == [pd.Timestamp("2025-09-01"), pd.Timestamp("2025-09-02 09:30:00")]
    assert result["change"].iloc[1] == pytest.approx(0.1)
