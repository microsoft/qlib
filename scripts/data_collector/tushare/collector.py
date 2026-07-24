from __future__ import annotations

import os
import sys
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import multiprocessing

import numpy as np
import pandas as pd
from loguru import logger
from qlib.utils import code_to_fname

try:
    import tushare as ts
except ImportError:  # pragma: no cover - optional dependency for tests without network
    ts = None

# ensure qlib scripts on path for relative imports
CUR_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = CUR_DIR.parent.parent
for p in (CUR_DIR, SCRIPTS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from dump_bin import DumpDataAll, DumpDataUpdate  # noqa: E402
from data_collector.base import BaseCollector, BaseNormalize, BaseRun, Normalize  # noqa: E402
from data_collector.utils import get_calendar_list  # noqa: E402

DEFAULT_BASE_DIR = CUR_DIR  # align with yahoo collector default_base_dir
DEFAULT_QLIB_DIR = Path.home() / ".qlib" / "qlib_data"


def _get_token() -> str:
    token = os.environ.get("TUSHARE_TOKEN")
    if not token:
        raise RuntimeError("TUSHARE_TOKEN is required; set it as an environment variable.")
    return token.strip()


def ts_code_to_qlib_symbol(ts_code: str) -> str:
    """Convert TuShare ts_code (e.g., 000001.SZ) to qlib symbol (e.g., sz000001)."""
    if not ts_code:
        return ts_code
    parts = ts_code.split(".")
    code = parts[0]
    suffix = parts[1].lower() if len(parts) > 1 else ""
    if suffix.startswith("sz"):
        return f"sz{code}"
    if suffix.startswith("sh"):
        return f"sh{code}"
    if suffix.startswith("bj"):
        return f"bj{code}"
    return f"{suffix}{code}" if suffix else code


def _normalize_factor(series: pd.Series) -> pd.Series:
    """Normalize adj_factor so the first valid value per symbol becomes 1.0."""
    if series.empty:
        return series
    first_valid = series.dropna().iloc[0] if series.dropna().size else np.nan
    if pd.isna(first_valid) or float(first_valid) == 0:
        return pd.Series([1.0] * len(series), index=series.index)
    return series / float(first_valid)


def normalize_tushare_eod(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize TuShare EOD dataframe to qlib-compatible CSV schema.

    Expected raw columns: ts_code, trade_date, open, high, low, close, vol, adj_factor[, amount]
    Output columns: date, open, high, low, close, volume, [amount], factor, symbol
    """
    if df is None or df.empty:
        return pd.DataFrame(
            columns=["date", "open", "high", "low", "close", "volume", "factor", "change", "symbol"]
        )

    data = df.copy()
    rename_map = {"trade_date": "date", "vol": "volume"}
    data.rename(columns=rename_map, inplace=True)

    if "date" not in data.columns:
        raise ValueError("Input dataframe must contain trade_date or date column.")

    # ensure yyyymmdd strings parsed correctly even if read as int
    data["date"] = pd.to_datetime(data["date"].astype(str))

    if "ts_code" in data.columns:
        data["symbol"] = data["ts_code"].apply(ts_code_to_qlib_symbol)
    elif "symbol" in data.columns:
        data["symbol"] = data["symbol"].apply(ts_code_to_qlib_symbol)
    else:
        raise ValueError("Input dataframe must contain ts_code or symbol column.")

    data.sort_values(["symbol", "date"], inplace=True)
    if "adj_factor" not in data.columns:
        data["adj_factor"] = 1.0
    data["adj_factor"] = data.groupby("symbol")["adj_factor"].transform(lambda s: s.ffill().bfill())
    data["factor"] = data.groupby("symbol")["adj_factor"].transform(_normalize_factor).fillna(1.0)

    for price_col in ["open", "high", "low", "close"]:
        if price_col in data.columns:
            data[price_col] = data[price_col].astype(float) * data["factor"]

    if "volume" in data.columns:
        safe_factor = data["factor"].replace({0: np.nan})
        data["volume"] = data["volume"].astype(float) / safe_factor

    cols = ["date", "open", "high", "low", "close", "volume", "factor", "symbol"]
    if "amount" in data.columns:
        data["amount"] = data["amount"].astype(float)
        cols.insert(cols.index("factor"), "amount")

    normalized = data[cols].copy()
    normalized["date"] = normalized["date"].dt.strftime("%Y-%m-%d")
    return normalized.reset_index(drop=True)


def dump_eod_to_qlib(
    data_path: Path,
    qlib_dir: Path,
    mode: str = "all",
    max_workers: int = 16,
    exclude_fields: str = "symbol,date",
    file_suffix: str = ".csv",
) -> Path:
    """
    Dump normalized EOD CSVs into qlib binary format.
    """
    qlib_dir = Path(qlib_dir).expanduser()
    qlib_dir.mkdir(parents=True, exist_ok=True)
    data_path = Path(data_path).expanduser()

    dumper_cls = DumpDataUpdate if mode.lower() == "update" else DumpDataAll
    dumper = dumper_cls(
        data_path=str(data_path),
        qlib_dir=str(qlib_dir),
        freq="day",
        max_workers=max_workers,
        date_field_name="date",
        symbol_field_name="symbol",
        exclude_fields=exclude_fields,
        file_suffix=file_suffix,
    )
    dumper.dump()
    return qlib_dir


def validate_qlib_dir(qlib_dir: Path, freq: str = "day") -> Dict[str, Optional[str]]:
    """
    Lightweight validation of a qlib directory. Returns a dict with None when healthy.
    """
    qlib_dir = Path(qlib_dir).expanduser()
    results: Dict[str, Optional[str]] = {"calendars": None, "instruments": None, "features": None}

    cal_file = qlib_dir / "calendars" / f"{freq}.txt"
    if not cal_file.exists() or cal_file.stat().st_size == 0:
        results["calendars"] = f"missing calendars at {cal_file}"

    inst_file = qlib_dir / "instruments" / "all.txt"
    if not inst_file.exists() or inst_file.stat().st_size == 0:
        results["instruments"] = f"missing instruments at {inst_file}"

    feat_dir = qlib_dir / "features"
    has_bins = feat_dir.exists() and any(feat_dir.glob("*/*.bin"))
    if not has_bins:
        results["features"] = f"no feature bins under {feat_dir}"

    return results


class TushareCollector(BaseCollector):
    """Daily TuShare collector following the data_collector.BaseCollector contract."""

    def __init__(
        self,
        save_dir: str | Path,
        start: Optional[str] = None,
        end: Optional[str] = None,
        interval: str = "1d",
        max_workers: int = 1,
        max_collector_count: int = 2,
        delay: float = 0,
        check_data_length: Optional[int] = None,
        limit_nums: Optional[int] = None,
        token: Optional[str] = None,
        pro_client=None,
        symbols: Optional[Iterable[str]] = None,
    ):
        if ts is None:
            raise ImportError("tushare is required; install it or add it to your venv.")
        self.token = token or _get_token()
        # avoid pickling non-serializable pro_client in multiprocessing; instantiate per call
        self._preset_symbols = list(symbols) if symbols else None
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

    def get_instrument_list(self) -> List[str]:
        if self._preset_symbols:
            return list(self._preset_symbols)
        pro = ts.pro_api(self.token)
        # include listed, delisted, paused to avoid survivor bias
        basic = pro.stock_basic(exchange="", list_status="L,D,P", fields="ts_code")
        return basic["ts_code"].dropna().unique().tolist()

    def normalize_symbol(self, symbol: str):
        return ts_code_to_qlib_symbol(symbol)

    def get_data(
        self, symbol: str, interval: str, start_datetime: pd.Timestamp, end_datetime: pd.Timestamp
    ) -> pd.DataFrame:
        if interval != self.INTERVAL_1d:
            raise ValueError("TushareCollector currently supports only 1d interval.")

        # determine incremental start based on existing csv to support resume
        start_dt = pd.Timestamp(start_datetime)
        end_dt = pd.Timestamp(end_datetime)

        symbol_fname = code_to_fname(self.normalize_symbol(symbol))
        existing_path = Path(self.save_dir).joinpath(f"{symbol_fname}.csv")
        last_date = None
        if existing_path.exists():
            try:
                # read minimal columns for efficiency
                existing = pd.read_csv(existing_path, usecols=lambda c: c in ["date", "trade_date"])
                if "date" in existing.columns:
                    existing["date"] = pd.to_datetime(existing["date"])
                    last_date = existing["date"].max()
                elif "trade_date" in existing.columns:
                    existing["trade_date"] = pd.to_datetime(existing["trade_date"])
                    last_date = existing["trade_date"].max()
            except Exception as e:  # pragma: no cover - best effort
                logger.warning(f"read existing csv failed for {symbol_fname}: {e}")

        if last_date is not None:
            start_dt = max(start_dt, last_date + pd.Timedelta(days=1))
        if start_dt >= end_dt:
            return pd.DataFrame()

        start_str = start_dt.strftime("%Y%m%d")
        end_str = end_dt.strftime("%Y%m%d")

        pro = ts.pro_api(self.token)
        daily = pro.daily(ts_code=symbol, start_date=start_str, end_date=end_str)
        adj = pro.adj_factor(ts_code=symbol, start_date=start_str, end_date=end_str)

        if daily is None or daily.empty:
            return pd.DataFrame()

        merged = pd.merge(daily, adj, on=["ts_code", "trade_date"], how="left")
        cols = ["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount", "adj_factor"]
        merged = merged[[c for c in cols if c in merged.columns]]
        merged["date"] = pd.to_datetime(merged["trade_date"])
        if last_date is not None:
            merged = merged[merged["date"] > last_date]
        return merged

    def save_instrument(self, symbol, df: pd.DataFrame):
        """
        Overwrite to avoid duplicate rows on rerun: always write deduped by date.
        """
        if df is None or df.empty:
            return

        df = df.copy()
        # ensure date column exists for dedup
        if "trade_date" in df.columns:
            df["date"] = pd.to_datetime(df["trade_date"])
        elif "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

        df["symbol"] = self.normalize_symbol(symbol)
        df.sort_values("date", inplace=True)
        df.drop_duplicates(subset=["date"], keep="last", inplace=True)
        if "trade_date" in df.columns:
            df.drop(columns=["trade_date"], inplace=True)

        symbol_fname = code_to_fname(df["symbol"].iloc[0])
        instrument_path = self.save_dir.joinpath(f"{symbol_fname}.csv")
        df.to_csv(instrument_path, index=False)


class TushareNormalize1d(BaseNormalize):
    """Normalize raw TuShare CSVs to qlib day-level format."""

    def _get_calendar_list(self) -> Iterable[pd.Timestamp]:
        token = os.environ.get("TUSHARE_TOKEN")
        if ts is not None and token:
            try:
                pro = ts.pro_api(token)
                today = pd.Timestamp.now().strftime("%Y%m%d")
                cal_df = pro.trade_cal(exchange="", start_date="20000101", end_date=today, fields="cal_date,is_open")
                cal_list = cal_df.loc[cal_df["is_open"] == 1, "cal_date"].map(pd.Timestamp).tolist()
                if cal_list:
                    return cal_list
            except Exception as e:  # pragma: no cover - network dependent
                logger.warning(f"TuShare trade_cal failed, fallback to default calendar: {e}")
        return get_calendar_list("ALL")

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        return normalize_tushare_eod(df)


class Run(BaseRun):
    collector_class_name = "TushareCollector"
    normalize_class_name = "TushareNormalize1d"
    default_base_dir = DEFAULT_BASE_DIR
    region = "CN"

    def __init__(
        self,
        source_dir=None,
        normalize_dir=None,
        max_workers: int = 1,
        interval: str = "1d",
        max_collector_count: int = 2,
    ):
        self.max_collector_count = max_collector_count
        super().__init__(source_dir=source_dir, normalize_dir=normalize_dir, max_workers=max_workers, interval=interval)

    def download_data(self, **kwargs):
        """
        Download raw TuShare daily data into source_dir.
        Pass token=..., symbols=..., start=..., end=... when needed.
        """
        return super().download_data(**kwargs)

    def normalize_data(self, date_field_name: str = "date", symbol_field_name: str = "symbol", **kwargs):
        """Normalize raw CSVs into factor-adjusted CSVs under normalize_dir."""
        return super().normalize_data(date_field_name=date_field_name, symbol_field_name=symbol_field_name, **kwargs)

    def dump_to_bin(
        self,
        qlib_dir: str | Path = DEFAULT_QLIB_DIR,
        mode: str = "all",
        max_workers: Optional[int] = None,
        exclude_fields: str = "symbol,date",
    ):
        """Dump normalized CSVs to qlib bin format."""
        workers = max_workers if max_workers is not None else self.max_workers
        return dump_eod_to_qlib(
            data_path=self.normalize_dir,
            qlib_dir=qlib_dir,
            mode=mode,
            max_workers=workers,
            exclude_fields=exclude_fields,
        )

    def download_today_data(
        self,
        max_collector_count=2,
        delay=0.5,
        check_data_length=None,
        limit_nums=None,
    ):
        """Download today's data (closed interval start, open interval end)."""
        start = pd.Timestamp.now().date()
        end = (pd.Timestamp(start) + pd.Timedelta(days=1)).date()
        return self.download_data(
            max_collector_count=max_collector_count,
            delay=delay,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            check_data_length=check_data_length,
            limit_nums=limit_nums,
        )

    def update_data_to_bin(
        self,
        qlib_data_1d_dir: str,
        end_date: str = None,
        check_data_length: int = None,
        delay: float = 1,
        exists_skip: bool = False,
    ):
        """
        Incrementally update an existing qlib dir using new TuShare data.
        """
        if self.interval.lower() != "1d":
            logger.warning("Currently only 1d interval incremental update is supported.")

        from qlib.utils import exists_qlib_data

        qlib_data_1d_dir = str(Path(qlib_data_1d_dir).expanduser().resolve())
        if not exists_qlib_data(qlib_data_1d_dir):
            raise RuntimeError(
                f"qlib_data_1d_dir not found or incomplete: {qlib_data_1d_dir}; "
                "build baseline with TuShare first (download_data -> normalize_data -> dump_to_bin), "
                "then rerun update_data_to_bin."
            )

        calendar_df = pd.read_csv(Path(qlib_data_1d_dir).joinpath("calendars/day.txt"))
        trading_date_ts = pd.Timestamp(calendar_df.iloc[-1, 0])
        # start from the last existing trading date; we only keep data strictly newer in dump
        trading_date = trading_date_ts.strftime("%Y-%m-%d")
        if end_date is None:
            end_date = (pd.Timestamp(trading_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        self.download_data(
            delay=delay,
            start=trading_date,
            end=end_date,
            check_data_length=check_data_length,
            max_collector_count=self.max_collector_count,
        )

        self.normalize_data()

        # 准备仅含增量日期的临时目录，减少 dump 工作量
        normalize_dir = Path(self.normalize_dir)
        inc_dir = normalize_dir.joinpath("__inc_tmp__")
        if inc_dir.exists():
            shutil.rmtree(inc_dir)
        inc_dir.mkdir(parents=True, exist_ok=True)

        last_date = trading_date_ts
        has_data = False
        for csv_file in normalize_dir.glob("*.csv"):
            df = pd.read_csv(csv_file)
            if "date" not in df.columns:
                continue
            df["date"] = pd.to_datetime(df["date"])
            df_new = df[df["date"] > last_date]
            if df_new.empty:
                continue
            has_data = True
            df_new.to_csv(inc_dir.joinpath(csv_file.name), index=False)

        if not has_data:
            shutil.rmtree(inc_dir, ignore_errors=True)
            logger.info("No incremental data found; skip dump.")
            return

        _dump = DumpDataUpdate(
            data_path=inc_dir,
            qlib_dir=qlib_data_1d_dir,
            exclude_fields="symbol,date",
            max_workers=self.max_workers,
        )
        _dump.dump()
        shutil.rmtree(inc_dir, ignore_errors=True)

        # Parse CN indices (CSI300/CSI100)
        try:
            from data_collector.cn_index.collector import get_instruments as get_cn_indices

            for _index in ["CSI100", "CSI300"]:
                get_cn_indices(str(qlib_data_1d_dir), _index, market_index="cn_index")
        except Exception as e:  # pragma: no cover - optional
            logger.warning(f"Index parsing skipped or failed: {e}")

    def pipeline(
        self,
        qlib_dir: str | Path = DEFAULT_QLIB_DIR,
        token: Optional[str] = None,
        symbols: Optional[Iterable[str]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ):
        """
        One-shot pipeline: download -> normalize -> dump.
        """
        self.download_data(token=token, symbols=symbols, start=start, end=end)
        self.normalize_data()
        return self.dump_to_bin(qlib_dir=qlib_dir)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    import fire

    fire.Fire(Run)

