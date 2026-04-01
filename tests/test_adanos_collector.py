import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))

from data_collector.adanos.collector import build_sentiment_frame, merge_price_and_sentiment


def test_build_sentiment_frame_aggregates_daily_source_rows():
    payloads = {
        "reddit": {
            "daily_trend": [
                {"date": "2026-03-25", "mentions": 10, "sentiment_score": 0.40, "buzz_score": 20.0},
            ]
        },
        "x": {
            "daily_trend": [
                {"date": "2026-03-25", "mentions": 20, "sentiment_score": 0.60, "buzz_score": 30.0, "avg_rank": 4.0},
            ]
        },
        "news": {
            "daily_trend": [
                {"date": "2026-03-25", "mentions": 5, "sentiment_score": 0.50, "buzz_score": 25.0},
            ]
        },
        "polymarket": {
            "daily_trend": [
                {"date": "2026-03-25", "trade_count": 100, "sentiment_score": 0.80, "buzz_score": 40.0},
            ]
        },
    }

    frame = build_sentiment_frame(
        symbol="aapl",
        payload_by_source=payloads,
        start_datetime=pd.Timestamp("2026-03-24"),
        end_datetime=pd.Timestamp("2026-03-26"),
    )

    assert len(frame) == 1
    row = frame.iloc[0]
    assert row["symbol"] == "AAPL"
    assert row["retail_buzz_avg"] == 28.75
    assert row["retail_sentiment_avg"] == 0.575
    assert row["retail_coverage"] == 4.0
    assert 0.0 <= row["retail_alignment_score"] <= 1.0
    assert row["x_avg_rank"] == 4.0
    assert row["polymarket_trade_count"] == 100


def test_merge_price_and_sentiment_keeps_price_rows():
    price_df = pd.DataFrame(
        [
            {"symbol": "AAPL", "date": "2026-03-24", "close": 100.0},
            {"symbol": "AAPL", "date": "2026-03-25", "close": 101.0},
        ]
    )
    sentiment_df = pd.DataFrame(
        [
            {"symbol": "AAPL", "date": "2026-03-25", "retail_buzz_avg": 28.75},
        ]
    )

    merged = merge_price_and_sentiment(price_df, sentiment_df)

    assert len(merged) == 2
    assert pd.isna(merged.loc[0, "retail_buzz_avg"])
    assert merged.loc[1, "retail_buzz_avg"] == 28.75


def test_build_sentiment_frame_handles_partially_missing_sources():
    payloads = {
        "reddit": {
            "daily_trend": [
                {"date": "2026-03-25", "mentions": 10, "sentiment_score": 0.40, "buzz_score": 20.0},
            ]
        },
        "x": {},
        "news": {},
        "polymarket": {},
    }

    frame = build_sentiment_frame(
        symbol="tsla",
        payload_by_source=payloads,
        start_datetime=pd.Timestamp("2026-03-24"),
        end_datetime=pd.Timestamp("2026-03-26"),
    )

    assert len(frame) == 1
    row = frame.iloc[0]
    assert row["symbol"] == "TSLA"
    assert row["retail_coverage"] == 1.0
    assert row["retail_alignment_score"] == 1.0
    assert pd.isna(row["x_buzz"])
