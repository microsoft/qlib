from __future__ import annotations

import csv
import datetime as dt
import json
import os
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import fire
except ImportError:  # pragma: no cover - optional dependency for environments without fire
    fire = None
import pandas as pd
import requests
from loguru import logger
from requests import Response

# ensure qlib scripts on path for relative imports (align with other collectors)
CUR_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = CUR_DIR.parent.parent
for p in (CUR_DIR, SCRIPTS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from dump_bin import DumpDataAll  # noqa: E402
from data_collector.base import BaseCollector, BaseNormalize, BaseRun, Normalize  # noqa: E402
from qlib.utils import code_to_fname  # noqa: E402


BINANCE_DATA_BASE = "https://data.binance.vision"
BINANCE_UM_EXCHANGE_INFO_URL = "https://fapi.binance.com/fapi/v1/exchangeInfo"
BINANCE_UM_KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"

DEFAULT_INST_PREFIX = "binance_um."

# Binance REST interval mapping
BINANCE_REST_INTERVAL_MAP = {
    "1min": "1m",
    "60min": "1h",
    "1d": "1d",
}

# Binance vision (public data) interval mapping (same strings as REST for these)
BINANCE_VISION_INTERVAL_MAP = {
    "1min": "1m",
    "60min": "1h",
    "1d": "1d",
}

# Qlib dump freq mapping
QLIB_FREQ_MAP = {
    "1min": "1min",
    "60min": "60min",
    "1d": "day",
}

RAW_KLINE_COLS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
]


@dataclass(frozen=True)
class Month:
    yyyy: int
    mm: int

    @classmethod
    def parse(cls, s: str) -> "Month":
        s = s.strip()
        if not s:
            raise ValueError("Empty month")
        if "-" in s:
            yyyy_s, mm_s = s.split("-", 1)
            return cls(int(yyyy_s), int(mm_s))
        if len(s) == 6:
            return cls(int(s[:4]), int(s[4:]))
        raise ValueError(f"Invalid month format: {s}. Use YYYY-MM or YYYYMM.")

    def to_ym(self) -> str:
        return f"{self.yyyy:04d}-{self.mm:02d}"


def _mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _parse_months(months: str) -> List[Month]:
    parts = [p.strip() for p in str(months).split(",") if p.strip()]
    if not parts:
        raise ValueError("months is empty; use '2023-11,2023-12' or '202311,202312'")
    return [Month.parse(p) for p in parts]


def _ts_to_utc_naive(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        return ts
    return ts.tz_convert("UTC").tz_localize(None)


def _to_ms(ts_like) -> int:
    ts = pd.Timestamp(ts_like)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return int(ts.timestamp() * 1000)


def _ms_to_utc_str(ms: int) -> str:
    return dt.datetime.utcfromtimestamp(ms / 1000.0).strftime("%Y-%m-%d %H:%M:%S")


def _read_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: Dict) -> None:
    _mkdir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp, path)


def get_um_perpetual_symbols(only_trading: bool = True, session: Optional[requests.Session] = None) -> List[str]:
    sess = session or requests.Session()
    resp = sess.get(BINANCE_UM_EXCHANGE_INFO_URL, timeout=30)
    if resp.status_code == 451:
        raise RuntimeError(
            "Binance Futures API is blocked in this network (HTTP 451). "
            "Please pass --symbols explicitly, or run behind an allowed network/VPN."
        )
    resp.raise_for_status()
    data = resp.json()
    symbols: List[str] = []
    for s in data.get("symbols", []):
        if s.get("contractType") != "PERPETUAL":
            continue
        if only_trading and s.get("status") != "TRADING":
            continue
        symbols.append(str(s.get("symbol")))
    symbols = sorted({x for x in symbols if x})
    return symbols


def build_um_monthly_zip_url(symbol: str, binance_interval: str, month: Month) -> str:
    # https://data.binance.vision/data/futures/um/monthly/klines/{SYMBOL}/{interval}/{SYMBOL}-{interval}-{YYYY-MM}.zip
    ym = month.to_ym()
    return (
        f"{BINANCE_DATA_BASE}/data/futures/um/monthly/klines/"
        f"{symbol}/{binance_interval}/{symbol}-{binance_interval}-{ym}.zip"
    )


def _safe_write_stream_to_file(resp: requests.Response, dst: Path, chunk_size: int = 1024 * 1024) -> None:
    tmp = dst.with_suffix(dst.suffix + ".part")
    _mkdir(dst.parent)
    with open(tmp, "wb") as f:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
    os.replace(tmp, dst)


def _iter_rows_from_zip(zip_path: Path) -> Iterable[List[str]]:
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise ValueError(f"No CSV inside zip: {zip_path}")
        csv_name = sorted(names)[0]
        with zf.open(csv_name, "r") as f:
            for raw_line in f:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                cols = line.split(",")
                # Binance public data CSV may include a header row in some exports.
                # Skip it safely.
                if cols and (cols[0] == "open_time" or not cols[0].isdigit()):
                    continue
                yield cols


def _normalize_kline_cols(cols: List[str]) -> Dict[str, float | int | str]:
    # Binance raw: [open_time, open, high, low, close, volume, close_time, quote_volume, trades, taker_buy_vol, taker_buy_quote, ignore]
    open_time_ms = int(cols[0])
    volume = float(cols[5])
    amount = float(cols[7])
    vwap = amount / volume if volume > 0 else float("nan")
    return {
        "date": _ms_to_utc_str(open_time_ms),
        "open": float(cols[1]),
        "high": float(cols[2]),
        "low": float(cols[3]),
        "close": float(cols[4]),
        "volume": volume,
        "amount": amount,
        "vwap": vwap,
        "trades": int(float(cols[8])),
        "taker_buy_volume": float(cols[9]),
        "taker_buy_amount": float(cols[10]),
    }


def _read_last_date_from_csv_fast(path: Path) -> Optional[pd.Timestamp]:
    """
    Read the last non-empty line and parse its 'date' column.
    This avoids loading large CSVs with pandas.
    """
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            # read tail window
            window = min(size, 256 * 1024)
            f.seek(-window, os.SEEK_END)
            tail = f.read(window).decode("utf-8", errors="ignore")
        lines = [ln for ln in tail.splitlines() if ln.strip()]
        if len(lines) < 2:
            return None
        header = lines[0].split(",")
        if "date" not in header:
            return None
        date_idx = header.index("date")
        last_line = lines[-1]
        parts = last_line.split(",")
        if len(parts) <= date_idx:
            return None
        return pd.Timestamp(parts[date_idx])
    except Exception:
        return None


class BinanceUMCollector(BaseCollector):
    """
    REST incremental collector (live) for Binance USDⓈ-M perpetual futures.
    """

    # extend BaseCollector to support 60min
    INTERVAL_60min = "60min"
    DEFAULT_START_DATETIME_60MIN = pd.Timestamp(dt.datetime.now() - pd.Timedelta(days=90)).date()
    DEFAULT_END_DATETIME_60MIN = BaseCollector.DEFAULT_END_DATETIME_1D

    def __init__(
        self,
        save_dir: str | Path,
        start=None,
        end=None,
        interval: str = "1min",
        max_workers: int = 1,
        max_collector_count: int = 2,
        delay: float = 0.2,
        check_data_length: int = None,
        limit_nums: int = None,
        symbols: Optional[str | List[str]] = None,
        include_non_trading: bool = False,
        inst_prefix: str = DEFAULT_INST_PREFIX,
        request_timeout: int = 30,
        request_limit: int = 1500,
    ):
        self._preset_symbols = self._parse_symbols(symbols)
        self.include_non_trading = include_non_trading
        self.inst_prefix = inst_prefix or ""
        self.request_timeout = int(request_timeout)
        self.request_limit = int(request_limit)
        self._session = requests.Session()
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
    def _parse_symbols(symbols: Optional[str | List[str]]) -> Optional[List[str]]:
        if symbols is None:
            return None
        if isinstance(symbols, (list, tuple)):
            out = [str(x).strip().upper() for x in symbols if str(x).strip()]
            return sorted(set(out)) or None
        s = str(symbols).strip()
        if not s:
            return None
        out = [p.strip().upper() for p in s.split(",") if p.strip()]
        return sorted(set(out)) or None

    def get_instrument_list(self) -> List[str]:
        if self._preset_symbols:
            return list(self._preset_symbols)
        return get_um_perpetual_symbols(only_trading=not self.include_non_trading, session=self._session)

    def normalize_symbol(self, symbol: str):
        sym = str(symbol).upper()
        return f"{self.inst_prefix}{sym}" if self.inst_prefix else sym

    def _get_existing_resume_start(self, symbol: str) -> Optional[pd.Timestamp]:
        """
        If per-symbol CSV exists, resume from (last_date + interval).
        """
        fname = code_to_fname(self.normalize_symbol(symbol))
        path = self.save_dir / f"{fname}.csv"
        last_dt = _read_last_date_from_csv_fast(path)
        if last_dt is None:
            return None
        if self.interval == self.INTERVAL_1min:
            return last_dt + pd.Timedelta(minutes=1)
        if self.interval == self.INTERVAL_60min:
            return last_dt + pd.Timedelta(hours=1)
        if self.interval == self.INTERVAL_1d:
            return last_dt + pd.Timedelta(days=1)
        return None

    def get_data(self, symbol: str, interval: str, start_datetime: pd.Timestamp, end_datetime: pd.Timestamp) -> pd.DataFrame:
        if interval not in BINANCE_REST_INTERVAL_MAP:
            raise ValueError(f"Unsupported interval={interval}. Supported: {sorted(BINANCE_REST_INTERVAL_MAP.keys())}")

        resume_start = self._get_existing_resume_start(symbol)
        start_dt = pd.Timestamp(start_datetime)
        if resume_start is not None and resume_start > start_dt:
            start_dt = resume_start

        end_dt = pd.Timestamp(end_datetime)
        if start_dt >= end_dt:
            return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "amount", "vwap", "trades"])

        binance_interval = BINANCE_REST_INTERVAL_MAP[interval]
        if interval == "1min":
            step_ms = 60_000
        elif interval == "60min":
            step_ms = 3_600_000
        else:
            step_ms = 86_400_000

        start_ms = _to_ms(start_dt)
        end_ms = _to_ms(end_dt)

        rows: List[Dict] = []
        while start_ms < end_ms:
            params = {
                "symbol": str(symbol).upper(),
                "interval": binance_interval,
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": self.request_limit,
            }
            self.sleep()
            try:
                resp: Response = self._session.get(BINANCE_UM_KLINES_URL, params=params, timeout=self.request_timeout)
                if resp.status_code == 451:
                    logger.warning(
                        "Binance Futures REST is blocked (HTTP 451). "
                        "This is a network/legal restriction. Skipping symbol={}."
                        .format(symbol)
                    )
                    return pd.DataFrame()
                resp.raise_for_status()
                data = resp.json()
            except requests.exceptions.RequestException as e:
                logger.warning(f"Binance REST request failed for {symbol} ({interval}): {e}")
                return pd.DataFrame()

            if not data:
                break

            last_open_time = None
            for cols in data:
                if not isinstance(cols, list) or len(cols) < 11:
                    continue
                k = _normalize_kline_cols([str(x) for x in cols])
                rows.append(k)
                last_open_time = int(cols[0])

            if last_open_time is None:
                break
            next_start = last_open_time + step_ms
            if next_start <= start_ms:
                # safety guard
                break
            start_ms = next_start

            # polite pacing for large symbols
            if self.delay and self.delay > 0:
                time.sleep(float(self.delay))

        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df["date"] = pd.to_datetime(df["date"])
        df = df.drop_duplicates(["date"]).sort_values(["date"]).reset_index(drop=True)
        if interval == "1d":
            df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        else:
            df["date"] = df["date"].dt.strftime("%Y-%m-%d %H:%M:%S")
        return df


class BinanceUMCollector1min(BinanceUMCollector):
    pass


class BinanceUMCollector60min(BinanceUMCollector):
    pass


class BinanceUMCollector1d(BinanceUMCollector):
    pass


class BinanceUMNormalize(BaseNormalize):
    def __init__(
        self,
        date_field_name: str = "date",
        symbol_field_name: str = "symbol",
        fill_missing: bool = True,
        fallback_1min_dir: Optional[str | Path] = None,
        **kwargs,
    ):
        self.fill_missing = bool(fill_missing)
        self.fallback_1min_dir = Path(fallback_1min_dir).expanduser().resolve() if fallback_1min_dir else None
        super().__init__(date_field_name=date_field_name, symbol_field_name=symbol_field_name, **kwargs)

    def _get_calendar_list(self) -> Iterable[pd.Timestamp]:
        # crypto is 24/7; we build calendar per instrument range in normalize()
        return []

    @staticmethod
    def _ensure_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        for c in cols:
            if c not in df.columns:
                df[c] = pd.NA
        return df

    @staticmethod
    def _compute_vwap(df: pd.DataFrame) -> pd.DataFrame:
        if "vwap" in df.columns:
            return df
        if "amount" in df.columns and "volume" in df.columns:
            vol = pd.to_numeric(df["volume"], errors="coerce")
            amt = pd.to_numeric(df["amount"], errors="coerce")
            df["vwap"] = amt / vol.replace({0: pd.NA})
        return df


class BinanceUMNormalize1min(BinanceUMNormalize):
    FREQ = "1min"

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=[self._date_field_name, self._symbol_field_name])

        data = df.copy()
        data = self._ensure_columns(
            data,
            [
                self._date_field_name,
                self._symbol_field_name,
                "open",
                "high",
                "low",
                "close",
                "volume",
                "amount",
                "vwap",
                "trades",
                "taker_buy_volume",
                "taker_buy_amount",
            ],
        )

        data[self._date_field_name] = pd.to_datetime(data[self._date_field_name], errors="coerce")
        data = data.dropna(subset=[self._date_field_name])
        data[self._date_field_name] = data[self._date_field_name].map(_ts_to_utc_naive)

        data = data.drop_duplicates([self._date_field_name]).sort_values([self._date_field_name]).reset_index(drop=True)
        data = self._compute_vwap(data)

        if self.fill_missing and not data.empty:
            start = data[self._date_field_name].min()
            end = data[self._date_field_name].max()
            full_idx = pd.date_range(start=start, end=end, freq=self.FREQ)
            data = data.set_index(self._date_field_name).reindex(full_idx)
            data.index.name = self._date_field_name
            # keep symbol stable
            sym = df[self._symbol_field_name].dropna().astype(str).iloc[0] if self._symbol_field_name in df.columns else ""
            data[self._symbol_field_name] = sym
            data = data.reset_index()

        data[self._date_field_name] = pd.to_datetime(data[self._date_field_name]).dt.strftime("%Y-%m-%d %H:%M:%S")
        return data


class BinanceUMNormalize60min(BinanceUMNormalize):
    FREQ = "60min"

    @staticmethod
    def _build_hourly_from_1min_df(df_1m: pd.DataFrame, date_col: str) -> pd.DataFrame:
        if df_1m is None or df_1m.empty:
            return pd.DataFrame()
        data = df_1m.copy()
        data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
        data = data.dropna(subset=[date_col])
        data = data.set_index(date_col).sort_index()

        agg = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "amount": "sum",
            "trades": "sum",
            "taker_buy_volume": "sum",
            "taker_buy_amount": "sum",
        }
        # Only aggregate columns that exist
        agg = {k: v for k, v in agg.items() if k in data.columns}
        out = data.resample("60min", label="left", closed="left").agg(agg)
        if "amount" in out.columns and "volume" in out.columns:
            out["vwap"] = out["amount"] / out["volume"].replace({0: pd.NA})
        out = out.dropna(how="all")
        out.reset_index(inplace=True)
        out[date_col] = out[date_col].map(_ts_to_utc_naive)
        return out

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=[self._date_field_name, self._symbol_field_name])

        data = df.copy()
        data = self._ensure_columns(
            data,
            [
                self._date_field_name,
                self._symbol_field_name,
                "open",
                "high",
                "low",
                "close",
                "volume",
                "amount",
                "vwap",
                "trades",
                "taker_buy_volume",
                "taker_buy_amount",
            ],
        )

        data[self._date_field_name] = pd.to_datetime(data[self._date_field_name], errors="coerce")
        data = data.dropna(subset=[self._date_field_name])
        data[self._date_field_name] = data[self._date_field_name].map(_ts_to_utc_naive)
        data = data.drop_duplicates([self._date_field_name]).sort_values([self._date_field_name]).reset_index(drop=True)
        data = self._compute_vwap(data)

        # optional fallback fill: use 1min -> 60min aggregation to fill missing hours
        if self.fallback_1min_dir is not None:
            sym = data[self._symbol_field_name].dropna().astype(str).iloc[0]
            fpath = self.fallback_1min_dir / f"{code_to_fname(sym)}.csv"
            if fpath.exists():
                try:
                    df_1m = pd.read_csv(fpath)
                    df_1h = self._build_hourly_from_1min_df(df_1m, self._date_field_name)
                    if not df_1h.empty:
                        df_1h[self._symbol_field_name] = sym
                        # align on date and fill empty values from aggregated
                        base = data.set_index(self._date_field_name)
                        fb = df_1h.set_index(self._date_field_name)
                        merged = base.combine_first(fb)
                        data = merged.reset_index()
                except Exception as e:
                    logger.warning(f"fallback_1min_dir aggregation failed for {sym}: {e}")

        if self.fill_missing and not data.empty:
            start = data[self._date_field_name].min()
            end = data[self._date_field_name].max()
            full_idx = pd.date_range(start=start, end=end, freq=self.FREQ)
            data = data.set_index(self._date_field_name).reindex(full_idx)
            data.index.name = self._date_field_name
            sym = df[self._symbol_field_name].dropna().astype(str).iloc[0] if self._symbol_field_name in df.columns else ""
            data[self._symbol_field_name] = sym
            data = data.reset_index()

        data[self._date_field_name] = pd.to_datetime(data[self._date_field_name]).dt.strftime("%Y-%m-%d %H:%M:%S")
        return data


class BinanceUMNormalize60minOnly(BinanceUMNormalize60min):
    pass


class BinanceUMNormalize1d(BinanceUMNormalize):
    FREQ = "1D"

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=[self._date_field_name, self._symbol_field_name])

        data = df.copy()
        data = self._ensure_columns(
            data,
            [
                self._date_field_name,
                self._symbol_field_name,
                "open",
                "high",
                "low",
                "close",
                "volume",
                "amount",
                "vwap",
                "trades",
                "taker_buy_volume",
                "taker_buy_amount",
            ],
        )

        data[self._date_field_name] = pd.to_datetime(data[self._date_field_name], errors="coerce")
        data = data.dropna(subset=[self._date_field_name])
        data[self._date_field_name] = data[self._date_field_name].map(_ts_to_utc_naive)

        # daily date should be normalized to date (no time)
        data[self._date_field_name] = pd.to_datetime(data[self._date_field_name]).dt.floor("D")
        data = data.drop_duplicates([self._date_field_name]).sort_values([self._date_field_name]).reset_index(drop=True)
        data = self._compute_vwap(data)

        if self.fill_missing and not data.empty:
            start = data[self._date_field_name].min()
            end = data[self._date_field_name].max()
            full_idx = pd.date_range(start=start, end=end, freq=self.FREQ)
            data = data.set_index(self._date_field_name).reindex(full_idx)
            data.index.name = self._date_field_name
            sym = df[self._symbol_field_name].dropna().astype(str).iloc[0] if self._symbol_field_name in df.columns else ""
            data[self._symbol_field_name] = sym
            data = data.reset_index()

        data[self._date_field_name] = pd.to_datetime(data[self._date_field_name]).dt.strftime("%Y-%m-%d")
        return data


class Run(BaseRun):
    """
    Qlib-style runner (fire CLI) for Binance UM perpetual futures.
    """

    def __init__(self, source_dir=None, normalize_dir=None, max_workers: int = 1, interval: str = "1min"):
        if interval not in QLIB_FREQ_MAP:
            raise ValueError(f"interval must be one of {sorted(QLIB_FREQ_MAP.keys())}")
        super().__init__(source_dir=source_dir, normalize_dir=normalize_dir, max_workers=max_workers, interval=interval)

    @property
    def collector_class_name(self):
        return f"BinanceUMCollector{self.interval}"

    @property
    def normalize_class_name(self):
        return f"BinanceUMNormalize{self.interval}"

    @property
    def default_base_dir(self) -> [Path, str]:
        return CUR_DIR

    # ---------- REST incremental ----------
    def download_data(
        self,
        max_collector_count: int = 2,
        delay: float = 0.2,
        start=None,
        end=None,
        check_data_length: int = None,
        limit_nums=None,
        symbols: Optional[str] = None,
        include_non_trading: bool = False,
        inst_prefix: str = DEFAULT_INST_PREFIX,
        request_timeout: int = 30,
        request_limit: int = 1500,
    ):
        """
        Download data from Binance REST (incremental, resume supported by reading existing CSV tail).

        Examples:
            python collector.py download_data --source_dir ~/.qlib/binance_um/source_1min --interval 1min --start 2024-01-01 --end 2024-02-01
            python collector.py download_data --source_dir ~/.qlib/binance_um/source_60min --interval 60min --start 2024-01-01 --end 2024-06-01
            python collector.py download_data --source_dir ~/.qlib/binance_um/source_1d --interval 1d --start 2024-01-01 --end 2024-06-01
        """
        if self.interval not in ("1min", "60min", "1d"):
            raise ValueError("Binance UM collector supports only 1min, 60min and 1d")
        return super().download_data(
            max_collector_count=max_collector_count,
            delay=delay,
            start=start,
            end=end,
            check_data_length=check_data_length,
            limit_nums=limit_nums,
            symbols=symbols,
            include_non_trading=include_non_trading,
            inst_prefix=inst_prefix,
            request_timeout=request_timeout,
            request_limit=request_limit,
        )

    # ---------- ZIP monthly history ----------
    def download_monthly_zip(
        self,
        months: str,
        raw_zip_dir: str | Path,
        zip_interval: str = "1m",
        symbols: Optional[str] = None,
        include_non_trading: bool = False,
        force: bool = False,
        sleep: float = 0.1,
    ):
        """
        Download Binance vision monthly zipped klines to raw_zip_dir.

        raw layout:
          raw_zip_dir/raw/um_perp/<interval>/<SYMBOL>/<SYMBOL>-<interval>-YYYY-MM.zip
          raw_zip_dir/manifest.json
        """
        months_list = _parse_months(months)
        raw_zip_dir = Path(raw_zip_dir).expanduser().resolve()
        _mkdir(raw_zip_dir)

        manifest_path = raw_zip_dir / "manifest.json"
        manifest = _read_json(manifest_path) or {"downloads": {}}

        sess = requests.Session()
        if symbols:
            sym_list = sorted({s.strip().upper() for s in symbols.split(",") if s.strip()})
        else:
            sym_list = get_um_perpetual_symbols(only_trading=not include_non_trading, session=sess)

        ok, miss, err, skip = 0, 0, 0, 0
        for sym in sym_list:
            for m in months_list:
                url = build_um_monthly_zip_url(sym, str(zip_interval), m)
                out_path = (
                    raw_zip_dir / "raw" / "um_perp" / str(zip_interval) / sym / f"{sym}-{zip_interval}-{m.to_ym()}.zip"
                )
                key = f"{sym}/{zip_interval}/{m.to_ym()}"
                prev = manifest.get("downloads", {}).get(key, {})
                if out_path.exists() and out_path.stat().st_size > 0 and prev.get("status") == "ok" and not force:
                    skip += 1
                    continue
                try:
                    resp = sess.get(url, stream=True, timeout=60)
                    if resp.status_code == 404:
                        miss += 1
                        manifest.setdefault("downloads", {})[key] = {
                            "status": "missing",
                            "url": url,
                            "path": str(out_path),
                            "ts": dt.datetime.utcnow().isoformat() + "Z",
                        }
                        _write_json(manifest_path, manifest)
                        time.sleep(float(sleep))
                        continue
                    if resp.status_code in (403, 451):
                        err += 1
                        manifest.setdefault("downloads", {})[key] = {
                            "status": "blocked",
                            "url": url,
                            "path": str(out_path),
                            "http_status": int(resp.status_code),
                            "ts": dt.datetime.utcnow().isoformat() + "Z",
                        }
                        _write_json(manifest_path, manifest)
                        time.sleep(float(sleep))
                        continue
                    resp.raise_for_status()
                    _safe_write_stream_to_file(resp, out_path)
                    size = out_path.stat().st_size
                    ok += 1
                    manifest.setdefault("downloads", {})[key] = {
                        "status": "ok",
                        "url": url,
                        "path": str(out_path),
                        "bytes": size,
                        "ts": dt.datetime.utcnow().isoformat() + "Z",
                    }
                    _write_json(manifest_path, manifest)
                except Exception as e:
                    err += 1
                    manifest.setdefault("downloads", {})[key] = {
                        "status": "error",
                        "url": url,
                        "path": str(out_path),
                        "error": str(e),
                        "ts": dt.datetime.utcnow().isoformat() + "Z",
                    }
                    _write_json(manifest_path, manifest)
                time.sleep(float(sleep))

        logger.info(f"download_monthly_zip done. ok={ok} miss={miss} err={err} skip={skip}")

    def convert_monthly_zip_to_source(
        self,
        raw_zip_dir: str | Path,
        zip_interval: str = "1m",
        source_dir: Optional[str | Path] = None,
        symbols: Optional[str] = None,
        inst_prefix: str = DEFAULT_INST_PREFIX,
        overwrite: bool = False,
    ):
        """
        Convert monthly zipped klines into per-symbol source CSVs (one file per instrument).

        Output files:
          <source_dir>/<binance_um.SYMBOL>.csv
        """
        raw_zip_dir = Path(raw_zip_dir).expanduser().resolve()
        src_dir = Path(source_dir).expanduser().resolve() if source_dir else self.source_dir
        _mkdir(src_dir)

        raw_root = raw_zip_dir / "raw" / "um_perp" / str(zip_interval)
        if not raw_root.exists():
            raise FileNotFoundError(f"raw zip root not found: {raw_root}")

        if symbols:
            sym_list = sorted({s.strip().upper() for s in symbols.split(",") if s.strip()})
        else:
            sym_list = sorted([p.name for p in raw_root.iterdir() if p.is_dir()])

        convert_manifest_path = raw_zip_dir / "convert_manifest.json"
        convert_manifest = _read_json(convert_manifest_path) or {"converted": {}}

        header = [
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
            "vwap",
            "trades",
            "taker_buy_volume",
            "taker_buy_amount",
            "symbol",
        ]

        for sym in sym_list:
            sym_dir = raw_root / sym
            if not sym_dir.exists():
                continue

            inst = f"{inst_prefix}{sym}" if inst_prefix else sym
            out_name = f"{code_to_fname(inst)}.csv"
            out_path = src_dir / out_name
            if overwrite and out_path.exists():
                out_path.unlink()

            # create if not exists
            if not out_path.exists():
                with open(out_path, "w", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(header)

            zip_paths = sorted(sym_dir.glob(f"{sym}-{zip_interval}-*.zip"))
            for zp in zip_paths:
                key = str(zp.resolve())
                if convert_manifest.get("converted", {}).get(key, {}).get("status") == "ok":
                    continue
                try:
                    rows_written = 0
                    with open(out_path, "a", newline="", encoding="utf-8") as f:
                        w = csv.writer(f)
                        for cols in _iter_rows_from_zip(zp):
                            if len(cols) < 11:
                                continue
                            k = _normalize_kline_cols(cols)
                            w.writerow(
                                [
                                    k["date"],
                                    k["open"],
                                    k["high"],
                                    k["low"],
                                    k["close"],
                                    k["volume"],
                                    k["amount"],
                                    k["vwap"],
                                    k["trades"],
                                    k["taker_buy_volume"],
                                    k["taker_buy_amount"],
                                    inst,
                                ]
                            )
                            rows_written += 1
                    convert_manifest.setdefault("converted", {})[key] = {
                        "status": "ok",
                        "out": str(out_path),
                        "rows": rows_written,
                        "ts": dt.datetime.utcnow().isoformat() + "Z",
                    }
                    _write_json(convert_manifest_path, convert_manifest)
                except Exception as e:
                    convert_manifest.setdefault("converted", {})[key] = {
                        "status": "error",
                        "out": str(out_path),
                        "error": str(e),
                        "ts": dt.datetime.utcnow().isoformat() + "Z",
                    }
                    _write_json(convert_manifest_path, convert_manifest)
                    logger.warning(f"convert failed: {zp} -> {out_path}: {e}")

        logger.info(f"convert_monthly_zip_to_source done. output_dir={src_dir}")

    # ---------- normalize ----------
    def normalize_data(
        self,
        date_field_name: str = "date",
        symbol_field_name: str = "symbol",
        fill_missing: bool = True,
        fallback_1min_dir: Optional[str | Path] = None,
        **kwargs,
    ):
        """
        Normalize per-symbol CSVs into Qlib-ready schema. For 60min, you can provide fallback_1min_dir to fill gaps.
        """
        _class = getattr(self._cur_module, self.normalize_class_name)
        normalizer = Normalize(
            source_dir=self.source_dir,
            target_dir=self.normalize_dir,
            normalize_class=_class,
            max_workers=self.max_workers,
            date_field_name=date_field_name,
            symbol_field_name=symbol_field_name,
            fill_missing=fill_missing,
            fallback_1min_dir=fallback_1min_dir,
            **kwargs,
        )
        normalizer.normalize()

    # ---------- dump ----------
    def dump_to_bin(
        self,
        qlib_dir: str | Path,
        max_workers: Optional[int] = None,
        exclude_fields: str = "symbol,date",
        file_suffix: str = ".csv",
    ):
        """
        Dump normalized CSVs to Qlib .bin using qlib/scripts/dump_bin.py's DumpDataAll.
        """
        qlib_dir = Path(qlib_dir).expanduser().resolve()
        _mkdir(qlib_dir)
        freq = QLIB_FREQ_MAP[self.interval]
        workers = int(max_workers) if max_workers is not None else int(self.max_workers)
        dumper = DumpDataAll(
            data_path=str(self.normalize_dir),
            qlib_dir=str(qlib_dir),
            freq=freq,
            max_workers=workers,
            date_field_name="date",
            symbol_field_name="symbol",
            exclude_fields=exclude_fields,
            file_suffix=file_suffix,
        )
        dumper.dump()
        logger.info(f"dump_to_bin done. qlib_dir={qlib_dir}")

    # ---------- optional: build 60min from 1min ----------
    def build_60min_from_1min(
        self,
        source_1min_dir: str | Path,
        target_60min_dir: str | Path,
        symbols: Optional[str] = None,
        overwrite: bool = False,
    ):
        """
        Aggregate 1min per-symbol CSVs into 60min per-symbol CSVs (source-stage).
        This is useful to fill gaps when hourly data is incomplete.
        """
        source_1min_dir = Path(source_1min_dir).expanduser().resolve()
        target_60min_dir = Path(target_60min_dir).expanduser().resolve()
        _mkdir(target_60min_dir)

        if symbols:
            sym_list = sorted({s.strip() for s in symbols.split(",") if s.strip()})
        else:
            sym_list = sorted([p.stem for p in source_1min_dir.glob("*.csv")])

        for stem in sym_list:
            in_path = source_1min_dir / f"{stem}.csv"
            if not in_path.exists():
                continue
            out_path = target_60min_dir / f"{stem}.csv"
            if out_path.exists() and not overwrite:
                continue
            try:
                df_1m = pd.read_csv(in_path)
                if df_1m.empty:
                    continue
                date_col = "date"
                sym_col = "symbol"
                sym_val = df_1m[sym_col].dropna().astype(str).iloc[0] if sym_col in df_1m.columns else stem
                df_1h = BinanceUMNormalize60min._build_hourly_from_1min_df(df_1m, date_col)
                if df_1h.empty:
                    continue
                df_1h[sym_col] = sym_val
                df_1h[date_col] = pd.to_datetime(df_1h[date_col]).dt.strftime("%Y-%m-%d %H:%M:%S")
                df_1h.to_csv(out_path, index=False)
            except Exception as e:
                logger.warning(f"build_60min_from_1min failed for {stem}: {e}")


if __name__ == "__main__":
    if fire is None:
        raise SystemExit(
            "Missing dependency: fire. Please install it in your venv, e.g.\n"
            "  pip install -r qlib/scripts/data_collector/binance_um/requirements.txt\n"
        )
    fire.Fire(Run)


