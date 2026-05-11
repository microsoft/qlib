# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
from pathlib import Path

import fire
import pandas as pd
from loguru import logger

CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent.parent))

from data_collector.base import BaseCollector, BaseNormalize, BaseRun, Normalize
from data_collector.utils import get_calendar_list

try:
    import akshare as ak
except ImportError:
    raise ImportError("Please install akshare: pip install akshare")

# A-share symbol suffix rules: SH for 6xx, SZ for others
SH_PREFIX = ("6",)
SZ_PREFIX = ("0", "3")

# B-share prefixes to exclude (USD/HKD denominated, not suitable for A-share workflows)
_B_PREFIXES = ("900", "200")


def get_all_symbols() -> list:
    """Auto-discover all A-share stock codes via akshare.

    Returns a sorted list of plain 6-digit codes (e.g. ['000001', '600519']).
    B-shares (900xxx, 200xxx) are excluded.
    """
    df = ak.stock_info_a_code_name()
    codes = df["code"].astype(str).str.zfill(6)
    codes = [c for c in codes if not c.startswith(_B_PREFIXES)]
    return sorted(set(codes))


def symbol_to_qlib(symbol: str) -> str:
    """Convert plain code like '000001' to Qlib format 'SZ000001'."""
    symbol = symbol.strip()
    if symbol.startswith(("sh", "sz", "SH", "SZ")):
        return symbol[:2].upper() + symbol[2:]
    if symbol.startswith(SH_PREFIX):
        return f"SH{symbol}"
    return f"SZ{symbol}"


def qlib_to_raw(symbol: str) -> str:
    """Convert 'SH600519' or 'SZ000001' to plain '600519' / '000001'."""
    symbol = symbol.strip().upper()
    for prefix in ("SH", "SZ"):
        if symbol.startswith(prefix):
            return symbol[len(prefix):]
    return symbol


FIELD_MAP = {
    "日期": "date",
    "股票代码": "symbol",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "money",
}


class AKShareCollector(BaseCollector):
    def __init__(
        self,
        save_dir,
        start=None,
        end=None,
        interval="1d",
        max_workers=1,
        max_collector_count=2,
        delay=0.5,
        check_data_length=None,
        limit_nums=None,
        symbols=None,
        symbol_file=None,
        adjust="qfq",
    ):
        self.requested_symbols = self._parse_symbols(symbols, symbol_file)
        self.adjust = adjust
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

    @staticmethod
    def _parse_symbols(symbols=None, symbol_file=None):
        result = []
        if symbols:
            if isinstance(symbols, str):
                result.extend(s.strip() for s in symbols.split(",") if s.strip())
            else:
                result.extend(str(s).strip() for s in symbols if str(s).strip())
        if symbol_file:
            path = Path(symbol_file).expanduser()
            result.extend(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        if not result:
            logger.info("No symbols provided, auto-discovering A-share universe via akshare...")
            result = get_all_symbols()
            logger.info(f"Discovered {len(result)} symbols")
        return sorted(set(result))

    def get_instrument_list(self):
        return self.requested_symbols

    def normalize_symbol(self, symbol: str):
        return symbol_to_qlib(symbol)

    def get_data(self, symbol, interval, start_datetime, end_datetime):
        raw = qlib_to_raw(symbol)
        start_str = pd.Timestamp(start_datetime).strftime("%Y%m%d")
        end_str = pd.Timestamp(end_datetime).strftime("%Y%m%d")

        try:
            df = ak.stock_zh_a_hist(
                symbol=raw,
                period="daily",
                start_date=start_str,
                end_date=end_str,
                adjust=self.adjust,
            )
        except Exception as e:
            logger.warning(f"AKShare fetch failed for {symbol}: {e}")
            return pd.DataFrame()

        if df is None or df.empty:
            return pd.DataFrame()

        df = df.rename(columns=FIELD_MAP)
        for col in ("open", "close", "high", "low", "volume", "money"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df["date"] = pd.to_datetime(df["date"])
        df["symbol"] = self.normalize_symbol(symbol)

        fields = ["date", "symbol", "open", "close", "high", "low", "volume", "money"]
        return df.loc[:, [f for f in fields if f in df.columns]]


class AKShareNormalize(BaseNormalize):
    def normalize(self, df):
        if df.empty:
            return df
        df = df.copy()
        df[self._date_field_name] = pd.to_datetime(df[self._date_field_name])
        df = df.drop_duplicates([self._date_field_name]).sort_values(self._date_field_name)
        return df

    def _get_calendar_list(self):
        return get_calendar_list("ALL")


class Run(BaseRun):
    def __init__(self, source_dir=None, normalize_dir=None, max_workers=1, interval="1d"):
        super().__init__(source_dir=source_dir, normalize_dir=normalize_dir, max_workers=max_workers, interval=interval)

    @property
    def collector_class_name(self):
        return "AKShareCollector"

    @property
    def normalize_class_name(self):
        return "AKShareNormalize"

    @property
    def default_base_dir(self):
        return CUR_DIR


if __name__ == "__main__":
    fire.Fire(Run)
