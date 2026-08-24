# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import re
import sys
from pathlib import Path
from typing import Iterable, List

import fire
import pandas as pd
import requests

CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent.parent))

from data_collector.base import BaseCollector, BaseNormalize, BaseRun


SEARCH_URL = "https://xquik.com/api/v1/x/tweets/search"
METRIC_FIELDS = {
    "bookmarkCount": "observed_bookmark_count",
    "likeCount": "observed_like_count",
    "quoteCount": "observed_quote_count",
    "replyCount": "observed_reply_count",
    "retweetCount": "observed_retweet_count",
    "viewCount": "observed_view_count",
}


def _as_utc(value, field_name: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be a valid timestamp") from error
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp


def _iso_z(timestamp: pd.Timestamp) -> str:
    return timestamp.isoformat().replace("+00:00", "Z")


def _parse_symbols(symbols) -> List[str]:
    values = symbols.split(",") if isinstance(symbols, str) else symbols
    if not isinstance(values, (list, tuple, set)):
        raise ValueError("symbols must be a comma-separated string or sequence")
    parsed = []
    for value in values:
        symbol = str(value).strip().lstrip("$").upper()
        if not symbol:
            continue
        if re.fullmatch(r"[A-Z0-9][A-Z0-9._-]{0,31}", symbol) is None:
            raise ValueError(f"invalid symbol: {symbol}")
        parsed.append(symbol)
    if not parsed:
        raise ValueError("symbols must contain at least one symbol")
    return sorted(set(parsed))


def _nonnegative_integer(value, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


class XquikCollector(BaseCollector):
    """Collect daily numeric X market-attention features from Xquik."""

    def __init__(
        self,
        save_dir,
        symbols,
        start=None,
        end=None,
        interval="1d",
        max_workers=1,
        max_collector_count=1,
        delay=0,
        check_data_length=None,
        limit_nums=None,
        query_template="${symbol}",
        language=None,
        include_replies=False,
        include_retweets=False,
        request_timeout=60,
    ):
        self._symbols = _parse_symbols(symbols)
        self._query_template = str(query_template)
        if "{symbol}" not in self._query_template:
            raise ValueError("query_template must contain {symbol}")
        self._language = str(language).strip() if language else None
        self._replies = "include" if include_replies else "exclude"
        self._retweets = "include" if include_retweets else "exclude"
        self._request_timeout = float(request_timeout)
        if self._request_timeout <= 0:
            raise ValueError("request_timeout must be greater than zero")
        self._api_key = os.getenv("X_TWITTER_SCRAPER_API_KEY", "").strip()
        if not self._api_key:
            raise ValueError("X_TWITTER_SCRAPER_API_KEY is required")

        super().__init__(
            save_dir=save_dir,
            start=start,
            end=end,
            interval=interval,
            max_workers=max_workers,
            max_collector_count=max_collector_count,
            delay=delay,
            check_data_length=check_data_length,
            limit_nums=limit_nums,
        )
        if self.interval != self.INTERVAL_1d:
            raise ValueError("Xquik collector supports only the 1d interval")
        if self.start_datetime >= self.end_datetime:
            raise ValueError("end must be later than start")
        if self.start_datetime != self.start_datetime.floor("D") or self.end_datetime != self.end_datetime.floor("D"):
            raise ValueError("start and end must be UTC day boundaries")

    def normalize_start_datetime(self, start_datetime=None):
        if start_datetime is None:
            raise ValueError("start is required")
        return _as_utc(start_datetime, "start")

    def normalize_end_datetime(self, end_datetime=None):
        if end_datetime is None:
            raise ValueError("end is required")
        return _as_utc(end_datetime, "end")

    def get_instrument_list(self):
        return self._symbols

    def normalize_symbol(self, symbol: str):
        return symbol.upper()

    def _request_day(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp):
        params = {
            "q": self._query_template.format(symbol=symbol),
            "mode": "coverage",
            "queryType": "Latest",
            "limit": 10000,
            "sinceTime": _iso_z(start),
            "untilTime": _iso_z(end),
            "replies": self._replies,
            "retweets": self._retweets,
        }
        if self._language:
            params["language"] = self._language
        response = requests.get(
            SEARCH_URL,
            headers={"x-api-key": self._api_key},
            params=params,
            timeout=self._request_timeout,
        )
        response.raise_for_status()
        payload = response.json()
        return self._validate_payload(payload)

    @staticmethod
    def _validate_payload(payload):
        if not isinstance(payload, dict):
            raise ValueError("Xquik response must be an object")
        tweets = payload.get("tweets")
        diagnostic = payload.get("diagnostic")
        if not isinstance(tweets, list) or not isinstance(diagnostic, dict):
            raise ValueError("Xquik coverage response must include tweets and diagnostic")

        required_flags = {
            "complete": True,
            "deadlineReached": False,
            "responseTruncated": False,
            "resultLimitReached": False,
        }
        for field_name, expected in required_flags.items():
            if diagnostic.get(field_name) is not expected:
                raise ValueError(f"Xquik coverage diagnostic {field_name} must be {expected}")
        for field_name in ("failedStrategyCount", "stalledStrategyCount"):
            if _nonnegative_integer(diagnostic.get(field_name), field_name) != 0:
                raise ValueError(f"Xquik coverage diagnostic {field_name} must be zero")
        returned_tweets = _nonnegative_integer(diagnostic.get("returnedTweets"), "returnedTweets")
        unique_tweets = _nonnegative_integer(diagnostic.get("uniqueTweets"), "uniqueTweets")
        if returned_tweets != len(tweets) or unique_tweets != len(tweets):
            raise ValueError("Xquik coverage counts do not match the returned tweets")
        return tweets

    @staticmethod
    def _aggregate_day(tweets, start: pd.Timestamp, end: pd.Timestamp):
        metric_totals = {field_name: 0 for field_name in METRIC_FIELDS.values()}
        tweet_ids = set()
        author_ids = set()
        for tweet in tweets:
            if not isinstance(tweet, dict):
                raise ValueError("each Xquik tweet must be an object")
            tweet_id = tweet.get("id")
            if not isinstance(tweet_id, str) or not tweet_id:
                raise ValueError("each Xquik tweet must have an ID")
            if tweet_id in tweet_ids:
                raise ValueError("Xquik coverage response contains duplicate tweet IDs")
            tweet_ids.add(tweet_id)

            created_at = tweet.get("createdAt")
            if not isinstance(created_at, str):
                raise ValueError("each Xquik tweet must have createdAt")
            created_timestamp = _as_utc(created_at, "createdAt")
            if not start <= created_timestamp < end:
                raise ValueError("Xquik returned a tweet outside the requested UTC day")

            author = tweet.get("author")
            if isinstance(author, dict) and isinstance(author.get("id"), str) and author["id"]:
                author_ids.add(author["id"])
            for response_name, feature_name in METRIC_FIELDS.items():
                metric_totals[feature_name] += _nonnegative_integer(tweet.get(response_name), response_name)

        return {
            "date": start.date().isoformat(),
            "post_count": len(tweet_ids),
            "author_count": len(author_ids),
            **metric_totals,
        }

    def get_data(self, symbol: str, interval: str, start_datetime: pd.Timestamp, end_datetime: pd.Timestamp):
        if interval != self.INTERVAL_1d:
            raise ValueError("Xquik collector supports only the 1d interval")
        rows = []
        day_start = start_datetime
        while day_start < end_datetime:
            day_end = min(day_start + pd.Timedelta(days=1), end_datetime)
            tweets = self._request_day(symbol, day_start, day_end)
            rows.append(self._aggregate_day(tweets, day_start, day_end))
            day_start = day_end
        observation_time = int(pd.Timestamp.now(tz="UTC").timestamp())
        for row in rows:
            row["observation_time"] = observation_time
        return pd.DataFrame(rows)


class XquikNormalize(BaseNormalize):
    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        normalized = df.copy()
        normalized[self._date_field_name] = pd.to_datetime(normalized[self._date_field_name], errors="raise")
        numeric_fields = set(normalized.columns) - {self._date_field_name, self._symbol_field_name}
        for field_name in numeric_fields:
            normalized[field_name] = pd.to_numeric(normalized[field_name], errors="raise")
        normalized.sort_values([self._date_field_name, "observation_time"], inplace=True)
        normalized.drop_duplicates([self._date_field_name, self._symbol_field_name], keep="last", inplace=True)
        return normalized.reset_index(drop=True)

    def _get_calendar_list(self) -> Iterable[pd.Timestamp]:
        return []


class XquikNormalize1d(XquikNormalize):
    pass


class Run(BaseRun):
    @property
    def collector_class_name(self):
        return "XquikCollector"

    @property
    def normalize_class_name(self):
        return f"XquikNormalize{self.interval}"

    @property
    def default_base_dir(self):
        return CUR_DIR


if __name__ == "__main__":
    fire.Fire(Run)
