# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import fire
import pandas as pd
import requests

CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent.parent))

from data_collector.base import BaseCollector, BaseNormalize, BaseRun, Normalize

DEFAULT_BASE_URL = "https://fxmacrodata.com/api/v1"
DEFAULT_PAIRS = (
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "AUDUSD",
    "USDCAD",
    "USDCHF",
    "NZDUSD",
)
DEFAULT_MACRO_CURRENCIES = ("usd",)
DEFAULT_MACRO_INDICATORS = (
    "inflation",
    "policy_rate",
    "unemployment",
    "non_farm_payrolls",
    "gdp",
)
API_KEY_ENV_VARS = ("FXMACRODATA_API_KEY", "FXMD_API_KEY")
OUTPUT_COLUMNS = [
    "date",
    "symbol",
    "open",
    "close",
    "high",
    "low",
    "volume",
    "factor",
    "change",
]
MACRO_DATASETS = ("announcements", "calendar", "predictions")
MACRO_OUTPUT_COLUMNS = [
    "date",
    "symbol",
    "value",
    "actual",
    "previous",
    "revised_previous",
    "consensus",
    "forecast",
    "surprise",
    "surprise_zscore",
    "prediction",
    "prediction_count",
    "announcement_datetime",
    "release_confirmed",
    "is_future",
]
MACRO_NUMERIC_COLUMNS = [col for col in MACRO_OUTPUT_COLUMNS if col not in {"date", "symbol"}]


class FXMacroDataCollector(BaseCollector):
    """Collect daily FX spot rates from FXMacroData."""

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
        pairs: [str, Sequence[str]] = None,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30,
    ):
        if interval != self.INTERVAL_1d:
            raise ValueError("FXMacroDataCollector supports daily data only: --interval 1d")

        self.pairs = self._normalize_pairs(pairs)
        self.api_key = api_key or self._get_env_api_key()
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        super(FXMacroDataCollector, self).__init__(
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
        return self.pairs

    def normalize_symbol(self, symbol: str):
        return self._normalize_pair(symbol).lower()

    def get_data(
        self,
        symbol: str,
        interval: str,
        start_datetime: pd.Timestamp,
        end_datetime: pd.Timestamp,
    ) -> pd.DataFrame:
        if interval != self.INTERVAL_1d:
            raise ValueError("FXMacroDataCollector supports daily data only: --interval 1d")

        pair = self._normalize_pair(symbol)
        base, quote = self._split_pair(pair)
        params = {
            "start_date": self._format_date(start_datetime),
            "end_date": self._format_date(end_datetime),
        }
        headers = {}
        if self.api_key:
            headers["X-API-Key"] = self.api_key

        response = requests.get(
            f"{self.base_url}/forex/{base}/{quote}",
            params=params,
            headers=headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        rows = self._payload_rows(response.json())
        return self._rows_to_frame(pair, rows)

    @classmethod
    def _normalize_pairs(cls, pairs: [str, Sequence[str], None]) -> List[str]:
        if pairs is None:
            return list(DEFAULT_PAIRS)
        if isinstance(pairs, str):
            pairs = [pair.strip() for pair in pairs.split(",") if pair.strip()]
        return [cls._normalize_pair(pair) for pair in pairs]

    @staticmethod
    def _normalize_pair(pair: str) -> str:
        pair = pair.strip().upper()
        if pair.endswith("=X"):
            pair = pair[:-2]
        pair = pair.replace("/", "").replace("-", "").replace("_", "")
        if len(pair) != 6 or not pair.isalpha():
            raise ValueError("FXMacroData pairs must look like EURUSD or EUR/USD")
        return pair

    @staticmethod
    def _split_pair(pair: str) -> Tuple[str, str]:
        return pair[:3].lower(), pair[3:].lower()

    @staticmethod
    def _get_env_api_key() -> Optional[str]:
        for name in API_KEY_ENV_VARS:
            value = os.getenv(name)
            if value:
                return value
        return None

    @staticmethod
    def _format_date(value: pd.Timestamp) -> str:
        return pd.Timestamp(value).strftime("%Y-%m-%d")

    @staticmethod
    def _payload_rows(payload) -> list:
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            data = payload.get("data", [])
            return data if isinstance(data, list) else []
        return []

    @classmethod
    def _rows_to_frame(cls, pair: str, rows: list) -> pd.DataFrame:
        records = []
        for row in rows:
            date = row.get("date") or row.get("timestamp")
            rate = cls._extract_rate(row)
            if date is None or rate is None:
                continue
            records.append(
                {
                    "date": pd.Timestamp(date),
                    "symbol": pair.lower(),
                    "open": rate,
                    "close": rate,
                    "high": rate,
                    "low": rate,
                    "volume": 0.0,
                    "factor": 1.0,
                }
            )
        if not records:
            return pd.DataFrame(columns=OUTPUT_COLUMNS)
        df = pd.DataFrame(records)
        df = df.drop_duplicates("date").sort_values("date").reset_index(drop=True)
        df["change"] = df["close"].ffill().pct_change().fillna(0.0)
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        return df[OUTPUT_COLUMNS]

    @staticmethod
    def _extract_rate(row: dict) -> Optional[float]:
        for key in ("val", "value", "close", "rate", "fx_rate"):
            value = row.get(key)
            if value is not None:
                return float(value)
        return None


class FXMacroDataMacroCollector(BaseCollector):
    """Collect macro announcements, release calendars, or forecasts."""

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
        dataset: str = "announcements",
        currencies: [str, Sequence[str]] = DEFAULT_MACRO_CURRENCIES,
        indicators: [str, Sequence[str]] = DEFAULT_MACRO_INDICATORS,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30,
    ):
        if interval != self.INTERVAL_1d:
            raise ValueError("FXMacroDataMacroCollector supports daily data only: --interval 1d")
        dataset = dataset.lower()
        if dataset not in MACRO_DATASETS:
            raise ValueError(f"dataset must be one of {MACRO_DATASETS}")

        self.dataset = dataset
        self.currencies = self._normalize_list(currencies, lower=True)
        self.indicators = self._normalize_list(indicators, lower=True)
        self.api_key = api_key or FXMacroDataCollector._get_env_api_key()
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        super(FXMacroDataMacroCollector, self).__init__(
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
        return [f"{currency}:{indicator}" for currency in self.currencies for indicator in self.indicators]

    def normalize_symbol(self, symbol: str):
        currency, indicator = self._split_macro_symbol(symbol)
        return f"{currency}_{indicator}"

    def get_data(
        self,
        symbol: str,
        interval: str,
        start_datetime: pd.Timestamp,
        end_datetime: pd.Timestamp,
    ) -> pd.DataFrame:
        if interval != self.INTERVAL_1d:
            raise ValueError("FXMacroDataMacroCollector supports daily data only: --interval 1d")

        currency, indicator = self._split_macro_symbol(symbol)
        params = {
            "start_date": FXMacroDataCollector._format_date(start_datetime),
            "end_date": FXMacroDataCollector._format_date(end_datetime),
        }

        if self.dataset == "announcements":
            rows = self._request_rows(f"announcements/{currency}/{indicator}", params)
        elif self.dataset == "calendar":
            rows = self._request_rows(f"calendar/{currency}", {**params, "indicator": indicator})
        else:
            rows = self._request_rows(f"predictions/{currency}/{indicator}", params)
        return self._rows_to_macro_frame(currency, indicator, rows)

    def _request_rows(self, path: str, params: dict) -> list:
        headers = {}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        response = requests.get(
            f"{self.base_url}/{path}",
            params={k: v for k, v in params.items() if v is not None},
            headers=headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return FXMacroDataCollector._payload_rows(response.json())

    @staticmethod
    def _normalize_list(values: [str, Sequence[str], None], lower=False) -> List[str]:
        if values is None:
            return []
        if isinstance(values, str):
            values = [value.strip() for value in values.split(",") if value.strip()]
        out = [str(value).strip() for value in values if str(value).strip()]
        return [value.lower() for value in out] if lower else out

    @staticmethod
    def _split_macro_symbol(symbol: str) -> Tuple[str, str]:
        normalized = symbol.strip().lower()
        if ":" in normalized:
            currency, indicator = normalized.split(":", 1)
        else:
            currency, indicator = normalized.split("_", 1)
        return currency, indicator

    @classmethod
    def _rows_to_macro_frame(cls, currency: str, indicator: str, rows: list) -> pd.DataFrame:
        records = []
        symbol = f"{currency}_{indicator}"
        for row in rows:
            date = row.get("date") or row.get("release_date")
            if date is None and row.get("announcement_datetime"):
                date = pd.to_datetime(row["announcement_datetime"], unit="s", utc=True)
            if date is None:
                continue
            prediction, prediction_count = cls._prediction_summary(row)
            actual = cls._number(row.get("actual"))
            value = cls._number(row.get("val") or row.get("value"))
            if value is None:
                value = actual
            records.append(
                {
                    "date": pd.Timestamp(date),
                    "symbol": symbol,
                    "value": value,
                    "actual": actual if actual is not None else value,
                    "previous": cls._number(row.get("previous")),
                    "revised_previous": cls._number(row.get("revised_previous")),
                    "consensus": cls._number(row.get("consensus") or row.get("expected") or row.get("estimate")),
                    "forecast": cls._number(row.get("forecast")),
                    "surprise": cls._number(row.get("surprise")),
                    "surprise_zscore": cls._number(row.get("surprise_zscore")),
                    "prediction": prediction,
                    "prediction_count": prediction_count,
                    "announcement_datetime": cls._int(row.get("announcement_datetime")),
                    "release_confirmed": 1.0 if row.get("release_date_confirmed") is True else 0.0,
                    "is_future": (
                        1.0
                        if row.get("announcement_timing") == "future" or row.get("actual_available") is False
                        else 0.0
                    ),
                }
            )
        if not records:
            return pd.DataFrame(columns=MACRO_OUTPUT_COLUMNS)
        df = pd.DataFrame(records)
        df = df.drop_duplicates(["date", "symbol"]).sort_values("date")
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        return df[MACRO_OUTPUT_COLUMNS]

    @classmethod
    def _prediction_summary(cls, row: dict) -> Tuple[Optional[float], float]:
        predictions = row.get("predictions")
        if isinstance(predictions, list) and predictions:
            prediction = cls._number(predictions[0].get("predicted_value"))
            return prediction, float(len(predictions))
        for key in ("forecast_prediction", "consensus_prediction"):
            prediction = row.get(key)
            if isinstance(prediction, dict):
                return cls._number(prediction.get("predicted_value")), 1.0
        return None, 0.0

    @staticmethod
    def _number(value) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _int(value) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None


class FXMacroDataNormalize(BaseNormalize):
    """Normalize FXMacroData CSVs for qlib dump_bin."""

    def _get_calendar_list(self) -> Iterable[pd.Timestamp]:
        return []

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        df = df.copy()
        df[self._date_field_name] = pd.to_datetime(df[self._date_field_name])
        df = df.drop_duplicates(self._date_field_name).sort_values(self._date_field_name)
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["close"])

        for column in ("open", "high", "low"):
            if column not in df.columns:
                df[column] = df["close"]
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(df["close"])
        df["volume"] = 0.0
        df["factor"] = 1.0
        df["change"] = df["close"].ffill().pct_change().fillna(0.0)
        df[self._date_field_name] = df[self._date_field_name].dt.strftime("%Y-%m-%d")
        df[self._symbol_field_name] = df[self._symbol_field_name].astype(str).str.lower()
        return df[OUTPUT_COLUMNS]


class FXMacroDataMacroNormalize(BaseNormalize):
    """Normalize FXMacroData macro features for qlib dump_bin."""

    def _get_calendar_list(self) -> Iterable[pd.Timestamp]:
        return []

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        df[self._date_field_name] = pd.to_datetime(df[self._date_field_name])
        df = df.drop_duplicates([self._date_field_name, self._symbol_field_name])
        df = df.sort_values([self._date_field_name, self._symbol_field_name])
        df[self._symbol_field_name] = df[self._symbol_field_name].astype(str).str.lower()
        for column in MACRO_NUMERIC_COLUMNS:
            if column not in df.columns:
                df[column] = None
            df[column] = pd.to_numeric(df[column], errors="coerce")
        df[self._date_field_name] = df[self._date_field_name].dt.strftime("%Y-%m-%d")
        return df[MACRO_OUTPUT_COLUMNS]


class Run(BaseRun):
    def download_data(
        self,
        max_collector_count=2,
        delay=0,
        start=None,
        end=None,
        check_data_length: int = None,
        limit_nums=None,
        pairs: str = ",".join(DEFAULT_PAIRS),
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30,
    ):
        """Download daily FX spot rates from FXMacroData."""

        super(Run, self).download_data(
            max_collector_count=max_collector_count,
            delay=delay,
            start=start,
            end=end,
            check_data_length=check_data_length,
            limit_nums=limit_nums,
            pairs=pairs,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )

    def download_macro_data(
        self,
        max_collector_count=2,
        delay=0,
        start=None,
        end=None,
        check_data_length: int = None,
        limit_nums=None,
        dataset: str = "announcements",
        currencies: str = ",".join(DEFAULT_MACRO_CURRENCIES),
        indicators: str = ",".join(DEFAULT_MACRO_INDICATORS),
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30,
    ):
        """Download FXMacroData announcements, calendars, or predictions."""

        FXMacroDataMacroCollector(
            self.source_dir,
            max_workers=self.max_workers,
            max_collector_count=max_collector_count,
            delay=delay,
            start=start,
            end=end,
            interval=self.interval,
            check_data_length=check_data_length,
            limit_nums=limit_nums,
            dataset=dataset,
            currencies=currencies,
            indicators=indicators,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        ).collector_data()

    def normalize_data(self, date_field_name: str = "date", symbol_field_name: str = "symbol"):
        """Normalize FXMacroData daily data."""

        if self.interval != "1d":
            raise ValueError("FXMacroData collector supports daily data only: --interval 1d")
        super(Run, self).normalize_data(date_field_name, symbol_field_name)

    def normalize_macro_data(self, date_field_name: str = "date", symbol_field_name: str = "symbol"):
        """Normalize FXMacroData macro feature CSVs."""

        Normalize(
            source_dir=self.source_dir,
            target_dir=self.normalize_dir,
            normalize_class=FXMacroDataMacroNormalize,
            max_workers=self.max_workers,
            date_field_name=date_field_name,
            symbol_field_name=symbol_field_name,
        ).normalize()

    @property
    def collector_class_name(self):
        return "FXMacroDataCollector"

    @property
    def normalize_class_name(self):
        return "FXMacroDataNormalize"

    @property
    def default_base_dir(self) -> [Path, str]:
        return CUR_DIR


if __name__ == "__main__":
    fire.Fire(Run)
