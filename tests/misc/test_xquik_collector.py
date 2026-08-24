# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
COLLECTOR_PATH = REPO_ROOT.joinpath("scripts/data_collector/xquik/collector.py")
sys.path.insert(0, str(REPO_ROOT.joinpath("scripts")))
SPEC = importlib.util.spec_from_file_location("xquik_collector", COLLECTOR_PATH)
xquik_collector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(xquik_collector)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def coverage_payload(tweets=None, **diagnostic_updates):
    tweets = [] if tweets is None else tweets
    diagnostic = {
        "complete": True,
        "deadlineReached": False,
        "failedStrategyCount": 0,
        "responseTruncated": False,
        "resultLimitReached": False,
        "returnedTweets": len(tweets),
        "stalledStrategyCount": 0,
        "uniqueTweets": len(tweets),
    }
    diagnostic.update(diagnostic_updates)
    return {"tweets": tweets, "diagnostic": diagnostic}


def tweet(tweet_id, created_at, author_id="author-1", **metrics):
    values = {
        "bookmarkCount": 1,
        "likeCount": 2,
        "quoteCount": 3,
        "replyCount": 4,
        "retweetCount": 5,
        "viewCount": 6,
    }
    values.update(metrics)
    return {
        "id": tweet_id,
        "createdAt": created_at,
        "author": {"id": author_id},
        **values,
    }


def make_collector(tmp_path, monkeypatch, **kwargs):
    monkeypatch.setenv("X_TWITTER_SCRAPER_API_KEY", "test-key")
    defaults = {
        "save_dir": tmp_path,
        "symbols": "msft,aapl,$AAPL",
        "start": "2026-08-01",
        "end": "2026-08-02",
    }
    defaults.update(kwargs)
    return xquik_collector.XquikCollector(**defaults)


def test_collects_daily_numeric_features(tmp_path, monkeypatch):
    calls = []
    tweets = [
        tweet("1", "2026-08-01T01:00:00Z", author_id="author-1"),
        tweet("2", "2026-08-01T22:00:00Z", author_id="author-2", likeCount=8, viewCount=10),
    ]

    def fake_get(url, headers, params, timeout):
        calls.append((url, headers, params, timeout))
        return FakeResponse(coverage_payload(tweets))

    monkeypatch.setattr(xquik_collector.requests, "get", fake_get)
    collector = make_collector(tmp_path, monkeypatch, symbols="aapl", language="en")
    result = collector.get_data("AAPL", "1d", collector.start_datetime, collector.end_datetime)

    assert result.drop(columns="observation_time").to_dict("records") == [
        {
            "date": "2026-08-01",
            "post_count": 2,
            "author_count": 2,
            "observed_bookmark_count": 2,
            "observed_like_count": 10,
            "observed_quote_count": 6,
            "observed_reply_count": 8,
            "observed_retweet_count": 10,
            "observed_view_count": 16,
        }
    ]
    assert result["observation_time"].iloc[0] > 0
    assert calls == [
        (
            xquik_collector.SEARCH_URL,
            {"x-api-key": "test-key"},
            {
                "q": "$AAPL",
                "mode": "coverage",
                "queryType": "Latest",
                "limit": 10000,
                "sinceTime": "2026-08-01T00:00:00Z",
                "untilTime": "2026-08-02T00:00:00Z",
                "replies": "exclude",
                "retweets": "exclude",
                "language": "en",
            },
            60.0,
        )
    ]


def test_emits_zero_row_for_complete_empty_day(tmp_path, monkeypatch):
    monkeypatch.setattr(xquik_collector.requests, "get", lambda *args, **kwargs: FakeResponse(coverage_payload()))
    collector = make_collector(tmp_path, monkeypatch, symbols="AAPL")

    result = collector.get_data("AAPL", "1d", collector.start_datetime, collector.end_datetime)

    assert result["post_count"].tolist() == [0]
    assert result["author_count"].tolist() == [0]
    assert result[list(xquik_collector.METRIC_FIELDS.values())].sum().sum() == 0


def test_writes_qlib_source_csv_without_post_content(tmp_path, monkeypatch):
    payload = coverage_payload([tweet("1", "2026-08-01T01:00:00Z")])
    monkeypatch.setattr(xquik_collector.requests, "get", lambda *args, **kwargs: FakeResponse(payload))
    collector = make_collector(tmp_path, monkeypatch, symbols="AAPL")

    collector._simple_collector("AAPL")

    saved = pd.read_csv(tmp_path.joinpath("AAPL.csv"))
    assert saved["symbol"].tolist() == ["AAPL"]
    assert saved["post_count"].tolist() == [1]
    assert "text" not in saved.columns
    assert "author" not in saved.columns
    assert "test-key" not in saved.to_csv(index=False)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("complete", False),
        ("deadlineReached", True),
        ("failedStrategyCount", 1),
        ("responseTruncated", True),
        ("resultLimitReached", True),
        ("stalledStrategyCount", 1),
    ],
)
def test_rejects_incomplete_coverage(tmp_path, monkeypatch, field_name, value):
    payload = coverage_payload(**{field_name: value})
    monkeypatch.setattr(xquik_collector.requests, "get", lambda *args, **kwargs: FakeResponse(payload))
    collector = make_collector(tmp_path, monkeypatch, symbols="AAPL")

    with pytest.raises(ValueError, match=field_name):
        collector.get_data("AAPL", "1d", collector.start_datetime, collector.end_datetime)


@pytest.mark.parametrize(
    "payload",
    [
        coverage_payload([tweet("1", "2026-08-01T01:00:00Z")], returnedTweets=0),
        coverage_payload([tweet("1", "2026-08-01T01:00:00Z")], uniqueTweets=0),
        coverage_payload([tweet("1", "2026-08-01T01:00:00Z"), tweet("1", "2026-08-01T02:00:00Z")]),
    ],
)
def test_rejects_inconsistent_coverage(tmp_path, monkeypatch, payload):
    monkeypatch.setattr(xquik_collector.requests, "get", lambda *args, **kwargs: FakeResponse(payload))
    collector = make_collector(tmp_path, monkeypatch, symbols="AAPL")

    with pytest.raises(ValueError):
        collector.get_data("AAPL", "1d", collector.start_datetime, collector.end_datetime)


@pytest.mark.parametrize(
    "invalid_tweet",
    [
        tweet("1", "2026-07-31T23:59:59Z"),
        tweet("1", "2026-08-02T00:00:00Z"),
        tweet("1", "not-a-timestamp"),
        tweet("1", "2026-08-01T01:00:00Z", likeCount=-1),
        tweet("1", "2026-08-01T01:00:00Z", likeCount=True),
    ],
)
def test_rejects_invalid_tweet_data(tmp_path, monkeypatch, invalid_tweet):
    payload = coverage_payload([invalid_tweet])
    monkeypatch.setattr(xquik_collector.requests, "get", lambda *args, **kwargs: FakeResponse(payload))
    collector = make_collector(tmp_path, monkeypatch, symbols="AAPL")

    with pytest.raises(ValueError):
        collector.get_data("AAPL", "1d", collector.start_datetime, collector.end_datetime)


def test_validates_configuration_before_collection(tmp_path, monkeypatch):
    monkeypatch.delenv("X_TWITTER_SCRAPER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="X_TWITTER_SCRAPER_API_KEY"):
        xquik_collector.XquikCollector(tmp_path, "AAPL", "2026-08-01", "2026-08-02")

    monkeypatch.setenv("X_TWITTER_SCRAPER_API_KEY", "test-key")
    with pytest.raises(ValueError, match="symbols"):
        xquik_collector.XquikCollector(tmp_path, "", "2026-08-01", "2026-08-02")
    with pytest.raises(ValueError, match="invalid symbol"):
        xquik_collector.XquikCollector(tmp_path, "../AAPL", "2026-08-01", "2026-08-02")
    with pytest.raises(ValueError, match="query_template"):
        xquik_collector.XquikCollector(tmp_path, "AAPL", "2026-08-01", "2026-08-02", query_template="stocks")
    with pytest.raises(ValueError, match="day boundaries"):
        xquik_collector.XquikCollector(tmp_path, "AAPL", "2026-08-01T12:00:00Z", "2026-08-02")
    with pytest.raises(ValueError, match="later"):
        xquik_collector.XquikCollector(tmp_path, "AAPL", "2026-08-02", "2026-08-01")
    with pytest.raises(ValueError, match="only the 1d"):
        xquik_collector.XquikCollector(tmp_path, "AAPL", "2026-08-01", "2026-08-02", interval="1min")


def test_normalizer_keeps_latest_observation():
    normalizer = xquik_collector.XquikNormalize1d()
    source = pd.DataFrame(
        [
            {"date": "2026-08-01", "symbol": "AAPL", "post_count": "2", "observation_time": "10"},
            {"date": "2026-08-01", "symbol": "AAPL", "post_count": "3", "observation_time": "20"},
            {"date": "2026-08-02", "symbol": "AAPL", "post_count": "1", "observation_time": "30"},
        ]
    )

    result = normalizer.normalize(source)

    assert result["post_count"].tolist() == [3, 1]
    assert result["observation_time"].tolist() == [20, 30]
    assert result["date"].tolist() == [pd.Timestamp("2026-08-01"), pd.Timestamp("2026-08-02")]
