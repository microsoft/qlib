# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import requests
from loguru import logger

CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent.parent))

from data_collector.base import BaseCollector, BaseNormalize, BaseRun  # noqa: E402


DEFAULT_BASE_URL = "https://api.adanos.org"
DEFAULT_MAX_LOOKBACK_DAYS = 90

SOURCE_CONFIG = {
    "reddit": {
        "endpoint": "/reddit/stocks/v1/stock/{symbol}",
        "trend_key": "daily_trend",
        "value_map": {
            "buzz_score": "reddit_buzz",
            "sentiment_score": "reddit_sentiment",
            "mentions": "reddit_mentions",
        },
    },
    "x": {
        "endpoint": "/x/stocks/v1/stock/{symbol}",
        "trend_key": "daily_trend",
        "value_map": {
            "buzz_score": "x_buzz",
            "sentiment_score": "x_sentiment",
            "mentions": "x_mentions",
            "avg_rank": "x_avg_rank",
        },
    },
    "news": {
        "endpoint": "/news/stocks/v1/stock/{symbol}",
        "trend_key": "daily_trend",
        "value_map": {
            "buzz_score": "news_buzz",
            "sentiment_score": "news_sentiment",
            "mentions": "news_mentions",
        },
    },
    "polymarket": {
        "endpoint": "/polymarket/stocks/v1/stock/{symbol}",
        "trend_key": "daily_trend",
        "value_map": {
            "buzz_score": "polymarket_buzz",
            "sentiment_score": "polymarket_sentiment",
            "trade_count": "polymarket_trade_count",
        },
    },
}

SOURCE_SENTIMENT_COLUMNS = [
    "reddit_sentiment",
    "x_sentiment",
    "news_sentiment",
    "polymarket_sentiment",
]
SOURCE_BUZZ_COLUMNS = [
    "reddit_buzz",
    "x_buzz",
    "news_buzz",
    "polymarket_buzz",
]


def _normalize_symbol(symbol: str) -> str:
    return str(symbol).strip().upper()


def _coerce_symbols(symbols: Optional[Sequence[str]]) -> List[str]:
    if symbols is None:
        return []
    if isinstance(symbols, str):
        symbols = [item.strip() for item in symbols.split(",")]
    return sorted({_normalize_symbol(item) for item in symbols if str(item).strip()})


def load_symbols_from_file(path: [str, Path]) -> List[str]:
    symbols = []
    with Path(path).expanduser().open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            token = line.replace(",", "\t").split("\t", 1)[0].strip()
            if token:
                symbols.append(_normalize_symbol(token))
    return sorted(set(symbols))


def resolve_api_lookback_days(start_datetime: pd.Timestamp, end_datetime: pd.Timestamp) -> int:
    days = max(int((pd.Timestamp(end_datetime) - pd.Timestamp(start_datetime)).days), 1)
    return min(days, DEFAULT_MAX_LOOKBACK_DAYS)


def _trend_rows_to_frame(source: str, payload: Optional[Dict]) -> pd.DataFrame:
    payload = payload or {}
    rows = payload.get(SOURCE_CONFIG[source]["trend_key"]) or []
    if not rows:
        return pd.DataFrame(columns=["date", *SOURCE_CONFIG[source]["value_map"].values()])

    frame = pd.DataFrame(rows)
    if "date" not in frame.columns:
        raise ValueError(f"Missing date field in {source} daily_trend payload")

    rename_map = SOURCE_CONFIG[source]["value_map"]
    frame = frame.rename(columns=rename_map)
    keep_columns = ["date", *rename_map.values()]
    frame = frame.loc[:, [column for column in keep_columns if column in frame.columns]]
    frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    return frame


def _alignment_score(values: Iterable[float]) -> float:
    values = [float(item) for item in values if pd.notna(item)]
    if len(values) <= 1:
        return 1.0 if values else 0.0
    center = sum(values) / len(values)
    mad = sum(abs(item - center) for item in values) / len(values)
    return float(max(0.0, min(1.0, 1.0 - mad / 2.0)))


def build_sentiment_frame(
    symbol: str,
    payload_by_source: Dict[str, Dict],
    start_datetime: pd.Timestamp,
    end_datetime: pd.Timestamp,
) -> pd.DataFrame:
    frames = []
    for source in SOURCE_CONFIG:
        source_frame = _trend_rows_to_frame(source, payload_by_source.get(source))
        if not source_frame.empty:
            frames.append(source_frame)

    if not frames:
        return pd.DataFrame()

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="date", how="outer")

    merged["date"] = pd.to_datetime(merged["date"])
    merged = merged.sort_values("date").drop_duplicates(["date"], keep="last")
    merged = merged[(merged["date"] >= pd.Timestamp(start_datetime)) & (merged["date"] < pd.Timestamp(end_datetime))]

    for source_config in SOURCE_CONFIG.values():
        for column in source_config["value_map"].values():
            if column not in merged.columns:
                merged[column] = pd.NA

    merged["retail_buzz_avg"] = merged[SOURCE_BUZZ_COLUMNS].mean(axis=1, skipna=True)
    merged["retail_sentiment_avg"] = merged[SOURCE_SENTIMENT_COLUMNS].mean(axis=1, skipna=True)
    merged["retail_coverage"] = merged[SOURCE_SENTIMENT_COLUMNS].notna().sum(axis=1).astype(float)
    merged["retail_alignment_score"] = merged[SOURCE_SENTIMENT_COLUMNS].apply(
        lambda row: _alignment_score(row.tolist()), axis=1
    )
    merged["symbol"] = _normalize_symbol(symbol)
    merged["date"] = merged["date"].dt.strftime("%Y-%m-%d")
    return merged.reset_index(drop=True)


def merge_price_and_sentiment(price_df: pd.DataFrame, sentiment_df: pd.DataFrame) -> pd.DataFrame:
    if price_df.empty:
        return price_df
    if sentiment_df.empty:
        return price_df.copy()

    merged = price_df.copy()
    merged["date"] = pd.to_datetime(merged["date"]).dt.strftime("%Y-%m-%d")
    merged["symbol"] = merged["symbol"].astype(str)

    sentiment = sentiment_df.copy()
    sentiment["date"] = pd.to_datetime(sentiment["date"]).dt.strftime("%Y-%m-%d")
    sentiment["symbol"] = sentiment["symbol"].astype(str)

    merged = merged.merge(sentiment, on=["symbol", "date"], how="left")
    return merged.sort_values(["date"]).reset_index(drop=True)


class AdanosCollector(BaseCollector):
    def __init__(
        self,
        save_dir: [str, Path],
        start=None,
        end=None,
        interval="1d",
        max_workers=1,
        max_collector_count=2,
        delay=0,
        check_data_length: int = None,
        limit_nums: int = None,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        symbols: Optional[Sequence[str]] = None,
        instruments_path: Optional[str] = None,
        timeout: int = 30,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.symbols = _coerce_symbols(symbols)
        self.instruments_path = instruments_path
        self.timeout = int(timeout)
        if not self.api_key:
            raise ValueError("api_key is required")
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

    def get_instrument_list(self):
        if self.symbols:
            return self.symbols
        if self.instruments_path:
            return load_symbols_from_file(self.instruments_path)
        raise ValueError("symbols or instruments_path is required")

    def normalize_symbol(self, symbol: str):
        return _normalize_symbol(symbol)

    def _request_source(self, source: str, symbol: str, days: int) -> Dict:
        url = f"{self.base_url}{SOURCE_CONFIG[source]['endpoint'].format(symbol=_normalize_symbol(symbol))}"
        response = requests.get(
            url,
            headers={"X-API-Key": self.api_key},
            params={"days": days},
            timeout=self.timeout,
        )
        if response.status_code == 404:
            logger.warning(f"{source} has no data for {symbol}")
            return {}
        response.raise_for_status()
        return response.json()

    def get_data(
        self, symbol: str, interval: str, start_datetime: pd.Timestamp, end_datetime: pd.Timestamp
    ) -> pd.DataFrame:
        if interval != self.INTERVAL_1d:
            raise ValueError(f"Adanos collector only supports {self.INTERVAL_1d}")

        days = resolve_api_lookback_days(start_datetime, end_datetime)
        payload_by_source = {}
        for source in SOURCE_CONFIG:
            try:
                payload_by_source[source] = self._request_source(source, symbol, days)
            except requests.HTTPError as error:
                logger.warning(f"{source} request failed for {symbol}: {error}")
            except requests.RequestException as error:
                logger.warning(f"{source} transport failed for {symbol}: {error}")

        return build_sentiment_frame(symbol, payload_by_source, start_datetime, end_datetime)


class AdanosCollector1d(AdanosCollector):
    pass


class AdanosNormalize(BaseNormalize):
    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        frame = df.copy()
        frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
        frame = frame.sort_values("date").drop_duplicates(["date"], keep="last")
        for column in frame.columns:
            if column not in {"date", "symbol"}:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        return frame.reset_index(drop=True)

    def _get_calendar_list(self) -> Iterable[pd.Timestamp]:
        return None


class AdanosNormalize1d(AdanosNormalize):
    pass


class Run(BaseRun):
    def __init__(self, source_dir=None, normalize_dir=None, max_workers=1, interval="1d"):
        super().__init__(source_dir, normalize_dir, max_workers, interval)

    @property
    def collector_class_name(self):
        return f"AdanosCollector{self.interval}"

    @property
    def normalize_class_name(self):
        return f"AdanosNormalize{self.interval}"

    @property
    def default_base_dir(self) -> [Path, str]:
        return CUR_DIR

    def download_data(
        self,
        api_key: str,
        symbols: Optional[str] = None,
        instruments_path: Optional[str] = None,
        max_collector_count=2,
        delay=0,
        start=None,
        end=None,
        check_data_length: int = None,
        limit_nums=None,
        timeout: int = 30,
    ):
        super().download_data(
            max_collector_count=max_collector_count,
            delay=delay,
            start=start,
            end=end,
            check_data_length=check_data_length,
            limit_nums=limit_nums,
            api_key=api_key,
            symbols=symbols,
            instruments_path=instruments_path,
            timeout=timeout,
        )

    def normalize_data(self, date_field_name: str = "date", symbol_field_name: str = "symbol"):
        super().normalize_data(date_field_name, symbol_field_name)

    def merge_with_price_data(
        self,
        price_dir: [str, Path],
        target_dir: [str, Path],
        sentiment_dir: Optional[str] = None,
        date_field_name: str = "date",
        symbol_field_name: str = "symbol",
    ):
        sentiment_dir = Path(sentiment_dir or self.normalize_dir).expanduser().resolve()
        price_dir = Path(price_dir).expanduser().resolve()
        target_dir = Path(target_dir).expanduser().resolve()
        target_dir.mkdir(parents=True, exist_ok=True)

        for price_path in sorted(price_dir.glob("*.csv")):
            price_df = pd.read_csv(price_path)
            sentiment_path = sentiment_dir.joinpath(price_path.name)
            if sentiment_path.exists():
                sentiment_df = pd.read_csv(sentiment_path)
            else:
                sentiment_df = pd.DataFrame(columns=[symbol_field_name, date_field_name])
            merged = merge_price_and_sentiment(price_df, sentiment_df)
            merged.to_csv(target_dir.joinpath(price_path.name), index=False)


if __name__ == "__main__":
    import fire

    fire.Fire(Run)
