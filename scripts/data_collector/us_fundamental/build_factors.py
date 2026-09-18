# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Build fundamental factors from Yahoo Finance data + SEC EDGAR filing dates.

This is the core pipeline of "Route 1.5":
    1. Read Yahoo fundamental data (quarterly financial statements)
    2. Read SEC EDGAR filing dates (when each 10-Q/10-K was actually filed)
    3. Use the filing date (NOT the report period date) as the availability date
    4. Compute fundamental factors (EP, BP, ROE, etc.)
    5. Forward-fill to daily frequency aligned with an existing Qlib calendar
    6. Output per-symbol CSVs ready for `dump_bin.py`

The key insight: a Q1 report (period ending 3/31) filed on 5/15 should only
be usable from 5/15 onwards. Using it from 4/1 would be look-ahead bias.

Usage:
    # Full pipeline
    python build_factors.py build \
        --yahoo_data_path ./yahoo_fundamental/_all_fundamentals.csv \
        --edgar_data_path ./edgar_filing_dates.csv \
        --qlib_dir ~/.qlib/qlib_data/us_data \
        --output_dir ./fundamental_daily

    # Then dump to Qlib binary format:
    python ../../dump_bin.py dump_all \
        --data_path ./fundamental_daily \
        --qlib_dir ~/.qlib/qlib_data/us_data \
        --freq day \
        --exclude_fields symbol,date

    # Alternative: if you have NO SEC EDGAR data, use a conservative lag
    python build_factors.py build \
        --yahoo_data_path ./yahoo_fundamental/_all_fundamentals.csv \
        --qlib_dir ~/.qlib/qlib_data/us_data \
        --output_dir ./fundamental_daily \
        --fallback_lag_days 90
"""

from pathlib import Path
from typing import Optional

import fire
import numpy as np
import pandas as pd
from loguru import logger


# ── Factor definitions ────────────────────────────────────────────────────────

def compute_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Compute fundamental factors from raw financial statement data.

    Input columns (from Yahoo Finance):
        TotalRevenue, GrossProfit, NetIncome, EBIT, EBITDA,
        TotalAssets, StockholdersEquity, TotalDebt,
        OperatingCashFlow, FreeCashFlow

    Output columns (added):
        gross_margin, roe, roa, accruals, debt_to_equity,
        revenue_yoy, earnings_yoy
    """
    df = df.copy()

    # ── Quality factors ──
    # Gross Margin
    df["gross_margin"] = df["GrossProfit"] / df["TotalRevenue"].replace(0, np.nan)

    # ROE = Net Income / Stockholders' Equity
    df["roe"] = df["NetIncome"] / df["StockholdersEquity"].replace(0, np.nan)

    # ROA = Net Income / Total Assets
    df["roa"] = df["NetIncome"] / df["TotalAssets"].replace(0, np.nan)

    # Accruals = (Net Income - Operating Cash Flow) / Total Assets
    # High accruals = low earnings quality
    df["accruals"] = (df["NetIncome"] - df["OperatingCashFlow"]) / df["TotalAssets"].replace(0, np.nan)

    # ── Leverage ──
    df["debt_to_equity"] = df["TotalDebt"] / df["StockholdersEquity"].replace(0, np.nan)

    # ── Growth factors (YOY) ──
    # Sort by symbol and date first for proper shift
    df = df.sort_values(["symbol", "reportDate"]).reset_index(drop=True)
    for col, out_col in [("TotalRevenue", "revenue_yoy"), ("NetIncome", "earnings_yoy")]:
        if col in df.columns:
            # YOY = current quarter vs same quarter last year (shift 4 quarters)
            df[out_col] = df.groupby("symbol")[col].transform(
                lambda x: x / x.shift(4).replace(0, np.nan) - 1
            )

    return df


def _merge_with_filing_dates(
    yahoo_df: pd.DataFrame,
    edgar_df: pd.DataFrame,
    fallback_lag_days: int = 90,
) -> pd.DataFrame:
    """Merge Yahoo fundamental data with SEC EDGAR filing dates.

    For each (symbol, reportDate) pair, find the corresponding filing date
    from SEC EDGAR. If no match is found, use reportDate + fallback_lag_days.

    Parameters
    ----------
    yahoo_df : pd.DataFrame
        Must have columns: [symbol, reportDate, ...]
    edgar_df : pd.DataFrame
        Must have columns: [symbol, filingDate, reportDate]
    fallback_lag_days : int
        Days to add to reportDate when no EDGAR match is found.

    Returns
    -------
    pd.DataFrame
        With added column 'availableDate': the date from which this data
        can be used without look-ahead bias.
    """
    yahoo_df = yahoo_df.copy()
    yahoo_df["reportDate"] = pd.to_datetime(yahoo_df["reportDate"])

    if edgar_df is not None and not edgar_df.empty:
        edgar_df = edgar_df.copy()
        edgar_df["filingDate"] = pd.to_datetime(edgar_df["filingDate"])
        edgar_df["reportDate"] = pd.to_datetime(edgar_df["reportDate"])

        # Merge on (symbol, reportDate)
        merged = yahoo_df.merge(
            edgar_df[["symbol", "reportDate", "filingDate"]],
            on=["symbol", "reportDate"],
            how="left",
        )

        # For unmatched rows, use conservative fallback
        no_match = merged["filingDate"].isna()
        if no_match.any():
            logger.info(
                f"{no_match.sum()}/{len(merged)} records have no EDGAR match, "
                f"using fallback lag of {fallback_lag_days} days"
            )
            merged.loc[no_match, "filingDate"] = (
                merged.loc[no_match, "reportDate"] + pd.Timedelta(days=fallback_lag_days)
            )
        merged["availableDate"] = merged["filingDate"]
    else:
        logger.warning(
            f"No EDGAR data provided. Using fallback lag of {fallback_lag_days} days "
            f"for all records. This may introduce minor look-ahead bias."
        )
        yahoo_df["availableDate"] = yahoo_df["reportDate"] + pd.Timedelta(days=fallback_lag_days)
        merged = yahoo_df

    return merged


def _forward_fill_to_daily(
    factor_df: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    factor_columns: list,
) -> pd.DataFrame:
    """Forward-fill quarterly factor data to daily frequency.

    For each symbol, at each calendar date, use the most recent factor values
    that were available (based on availableDate, not reportDate).

    Parameters
    ----------
    factor_df : pd.DataFrame
        Must have columns: [symbol, availableDate] + factor_columns
    calendar : pd.DatetimeIndex
        The trading calendar to align to.
    factor_columns : list of str
        Which columns to forward-fill.

    Returns
    -------
    pd.DataFrame
        Daily data with columns: [date, symbol] + factor_columns
    """
    all_daily = []
    symbols = factor_df["symbol"].unique()

    for symbol in symbols:
        sym_df = factor_df[factor_df["symbol"] == symbol].copy()
        sym_df = sym_df.sort_values("availableDate").drop_duplicates("availableDate", keep="last")

        # Create a daily series using the calendar
        daily = pd.DataFrame({"date": calendar})
        daily["symbol"] = symbol

        # For each factor column, forward-fill from availableDate
        for col in factor_columns:
            if col not in sym_df.columns:
                daily[col] = np.nan
                continue

            # Build a series indexed by availableDate
            values = sym_df.set_index("availableDate")[col]
            values = values[~values.index.duplicated(keep="last")]

            # Reindex to calendar and forward-fill
            aligned = values.reindex(calendar, method="ffill")
            daily[col] = aligned.values

        all_daily.append(daily)

    if not all_daily:
        return pd.DataFrame()

    result = pd.concat(all_daily, ignore_index=True)
    return result


def build(
    yahoo_data_path: str,
    output_dir: str,
    edgar_data_path: Optional[str] = None,
    qlib_dir: Optional[str] = None,
    calendar_path: Optional[str] = None,
    fallback_lag_days: int = 90,
    start: Optional[str] = None,
    end: Optional[str] = None,
):
    """Build daily fundamental factor CSVs from Yahoo + EDGAR data.

    Parameters
    ----------
    yahoo_data_path : str
        Path to Yahoo fundamental CSV (output of yahoo_fundamental.py).
    output_dir : str
        Directory to save per-symbol daily CSVs (input to dump_bin.py).
    edgar_data_path : str, optional
        Path to EDGAR filing dates CSV (output of edgar_filing_dates.py).
        If None, uses conservative fallback lag.
    qlib_dir : str, optional
        Path to existing Qlib data directory (to read trading calendar).
        Either qlib_dir or calendar_path must be provided.
    calendar_path : str, optional
        Path to calendar file (one date per line). Overrides qlib_dir.
    fallback_lag_days : int
        Days to add to reportDate when no EDGAR filing date is available.
        Default 90 (conservative: SEC requires 10-Q within 40-45 days).
    start : str, optional
        Start date filter for output data.
    end : str, optional
        End date filter for output data.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load data ──
    logger.info("Loading Yahoo fundamental data...")
    yahoo_df = pd.read_csv(yahoo_data_path)
    yahoo_df["reportDate"] = pd.to_datetime(yahoo_df["reportDate"])
    logger.info(f"  {len(yahoo_df)} records, {yahoo_df['symbol'].nunique()} symbols")

    edgar_df = None
    if edgar_data_path and Path(edgar_data_path).exists():
        logger.info("Loading SEC EDGAR filing dates...")
        edgar_df = pd.read_csv(edgar_data_path)
        logger.info(f"  {len(edgar_df)} filing records")

    # ── Step 2: Load calendar ──
    if calendar_path:
        calendar = pd.to_datetime(
            pd.read_csv(calendar_path, header=None)[0]
        )
    elif qlib_dir:
        cal_file = Path(qlib_dir) / "calendars" / "day.txt"
        if not cal_file.exists():
            raise FileNotFoundError(f"Calendar not found: {cal_file}")
        calendar = pd.to_datetime(
            pd.read_csv(cal_file, header=None)[0]
        )
    else:
        raise ValueError("Must provide either qlib_dir or calendar_path")

    if start:
        calendar = calendar[calendar >= pd.Timestamp(start)]
    if end:
        calendar = calendar[calendar <= pd.Timestamp(end)]
    calendar = pd.DatetimeIndex(sorted(calendar))
    logger.info(f"Calendar: {calendar[0].date()} to {calendar[-1].date()}, {len(calendar)} days")

    # ── Step 3: Merge with filing dates ──
    logger.info("Merging with filing dates...")
    merged = _merge_with_filing_dates(yahoo_df, edgar_df, fallback_lag_days)

    # ── Step 4: Compute factors ──
    logger.info("Computing fundamental factors...")
    factor_df = compute_factors(merged)

    # Factor columns to output (these will become Qlib features)
    factor_columns = [
        # Raw values (for computing price-relative factors in handler)
        "NetIncome",
        "TotalRevenue",
        "StockholdersEquity",
        "TotalAssets",
        "TotalDebt",
        "OperatingCashFlow",
        "FreeCashFlow",
        "EBITDA",
        # Computed factors
        "gross_margin",
        "roe",
        "roa",
        "accruals",
        "debt_to_equity",
        "revenue_yoy",
        "earnings_yoy",
    ]
    # Only keep columns that actually exist
    factor_columns = [c for c in factor_columns if c in factor_df.columns]

    # ── Step 5: Forward-fill to daily ──
    logger.info("Forward-filling to daily frequency...")
    daily_df = _forward_fill_to_daily(factor_df, calendar, factor_columns)
    logger.info(f"Daily data: {len(daily_df)} rows, {daily_df['symbol'].nunique()} symbols")

    # ── Step 6: Save per-symbol CSVs ──
    logger.info(f"Saving to {output_dir}...")
    # Rename columns to lowercase for Qlib convention
    rename_map = {c: c.lower() for c in factor_columns if c != c.lower()}
    daily_df.rename(columns=rename_map, inplace=True)
    factor_columns_lower = [c.lower() for c in factor_columns]

    saved_count = 0
    for symbol, sym_df in daily_df.groupby("symbol"):
        # Drop rows where ALL factors are NaN (before first filing)
        sym_df = sym_df.dropna(subset=factor_columns_lower, how="all")
        if sym_df.empty:
            continue
        sym_df.to_csv(output_dir / f"{symbol}.csv", index=False)
        saved_count += 1

    logger.info(f"Saved {saved_count} symbol files to {output_dir}")
    logger.info(
        f"\nNext step: dump to Qlib binary format:\n"
        f"  python scripts/dump_bin.py dump_all \\\n"
        f"    --data_path {output_dir} \\\n"
        f"    --qlib_dir <your_qlib_data_dir> \\\n"
        f"    --freq day \\\n"
        f"    --exclude_fields symbol,date\n"
        f"\n"
        f"  NOTE: Use dump_update instead of dump_all if you want to ADD\n"
        f"  fundamental features to an existing Qlib dataset that already\n"
        f"  has OHLCV data."
    )


if __name__ == "__main__":
    fire.Fire({"build": build})
