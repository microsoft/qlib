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

from data_collector.base import BaseCollector, BaseNormalize, BaseRun

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
            raise ValueError(
                "FXMacroDataCollector supports daily data only: --interval 1d"
            )

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
            raise ValueError(
                "FXMacroDataCollector supports daily data only: --interval 1d"
            )

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
            raise ValueError(
                "FXMacroData pairs must look like EURUSD or EUR/USD"
            )
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
        df = (
            df.drop_duplicates("date")
            .sort_values("date")
            .reset_index(drop=True)
        )
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


class FXMacroDataNormalize(BaseNormalize):
    """Normalize FXMacroData CSVs for qlib dump_bin."""

    def _get_calendar_list(self) -> Iterable[pd.Timestamp]:
        return []

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        df = df.copy()
        df[self._date_field_name] = pd.to_datetime(df[self._date_field_name])
        df = df.drop_duplicates(self._date_field_name).sort_values(
            self._date_field_name
        )
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["close"])

        for column in ("open", "high", "low"):
            if column not in df.columns:
                df[column] = df["close"]
            df[column] = pd.to_numeric(
                df[column], errors="coerce"
            ).fillna(df["close"])
        df["volume"] = 0.0
        df["factor"] = 1.0
        df["change"] = df["close"].ffill().pct_change().fillna(0.0)
        df[self._date_field_name] = df[self._date_field_name].dt.strftime(
            "%Y-%m-%d"
        )
        df[self._symbol_field_name] = (
            df[self._symbol_field_name].astype(str).str.lower()
        )
        return df[OUTPUT_COLUMNS]


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

    def normalize_data(
        self, date_field_name: str = "date", symbol_field_name: str = "symbol"
    ):
        """Normalize FXMacroData daily data."""

        if self.interval != "1d":
            raise ValueError(
                "FXMacroData collector supports daily data only: --interval 1d"
            )
        super(Run, self).normalize_data(date_field_name, symbol_field_name)

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
