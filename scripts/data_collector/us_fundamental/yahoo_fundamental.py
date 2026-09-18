# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Collect fundamental data from Yahoo Finance for US stocks.

This module uses yahooquery (already a Qlib dependency) to fetch quarterly
and annual financial statements (income statement, balance sheet, cash flow).
From these raw statements, we compute standard fundamental factors:

    Value:   EP, BP, SP, CFP
    Quality: ROE, ROA, GrossMargin, Accruals
    Growth:  RevenueYOY, EarningsYOY
    Leverage: DebtToEquity

Usage:
    python yahoo_fundamental.py collect \
        --symbols AAPL,MSFT,GOOGL \
        --save_dir ./yahoo_fundamental \
        --start 2018-01-01

    python yahoo_fundamental.py collect_from_file \
        --symbol_file ./symbols.txt \
        --save_dir ./yahoo_fundamental
"""

import time
from pathlib import Path
from typing import List, Optional, Union

import fire
import numpy as np
import pandas as pd
from loguru import logger
from yahooquery import Ticker


# ── Fields we extract from Yahoo Finance ──────────────────────────────────────
# Income statement fields
INCOME_FIELDS = [
    "TotalRevenue",
    "GrossProfit",
    "NetIncome",
    "EBIT",
    "EBITDA",
    "CostOfRevenue",
]

# Balance sheet fields
BALANCE_FIELDS = [
    "TotalAssets",
    "StockholdersEquity",
    "TotalDebt",
    "CurrentAssets",
    "CurrentLiabilities",
]

# Cash flow fields
CASHFLOW_FIELDS = [
    "OperatingCashFlow",
    "FreeCashFlow",
    "CapitalExpenditure",
]


def _safe_get_financial(ticker_obj: Ticker, method: str, frequency: str = "q") -> pd.DataFrame:
    """Safely call a yahooquery financial method.

    Returns an empty DataFrame if the call fails or returns a dict (error).
    """
    try:
        func = getattr(ticker_obj, method)
        result = func(frequency=frequency)
        if isinstance(result, pd.DataFrame) and not result.empty:
            return result
        return pd.DataFrame()
    except Exception as e:
        logger.debug(f"Failed to get {method}: {e}")
        return pd.DataFrame()


def _collect_single_symbol(symbol: str, start: Optional[str] = None) -> pd.DataFrame:
    """Collect fundamental data for a single symbol.

    Returns a DataFrame with columns: [date, symbol, field1, field2, ...]
    where each row is a quarterly snapshot.
    """
    ticker = Ticker(symbol, asynchronous=False)

    # Collect quarterly financial data
    income_df = _safe_get_financial(ticker, "income_statement", "q")
    balance_df = _safe_get_financial(ticker, "balance_sheet", "q")
    cashflow_df = _safe_get_financial(ticker, "cash_flow", "q")

    if income_df.empty and balance_df.empty and cashflow_df.empty:
        logger.warning(f"{symbol}: no financial data available")
        return pd.DataFrame()

    # Normalize index: yahooquery returns MultiIndex (symbol, asOfDate)
    dfs = {}
    for name, df, fields in [
        ("income", income_df, INCOME_FIELDS),
        ("balance", balance_df, BALANCE_FIELDS),
        ("cashflow", cashflow_df, CASHFLOW_FIELDS),
    ]:
        if df.empty:
            continue
        # Reset index to get asOfDate as column
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index()
        # Standardize date column
        if "asOfDate" in df.columns:
            df["asOfDate"] = pd.to_datetime(df["asOfDate"])
        elif "index" in df.columns:
            df.rename(columns={"index": "asOfDate"}, inplace=True)
            df["asOfDate"] = pd.to_datetime(df["asOfDate"])

        # Select only the fields we care about
        available_fields = [f for f in fields if f in df.columns]
        if not available_fields:
            continue

        keep_cols = ["asOfDate"] + available_fields
        df = df[keep_cols].copy()
        df = df.drop_duplicates("asOfDate").sort_values("asOfDate")
        dfs[name] = df

    if not dfs:
        return pd.DataFrame()

    # Merge all financial data on asOfDate
    merged = None
    for df in dfs.values():
        if merged is None:
            merged = df
        else:
            merged = pd.merge(merged, df, on="asOfDate", how="outer")

    merged = merged.sort_values("asOfDate").reset_index(drop=True)
    merged["symbol"] = symbol
    merged.rename(columns={"asOfDate": "reportDate"}, inplace=True)

    if start:
        merged = merged[merged["reportDate"] >= pd.Timestamp(start)]

    return merged


def collect_fundamental_data(
    symbols: Union[str, List[str]],
    save_dir: Optional[str] = None,
    start: Optional[str] = None,
    delay: float = 0.5,
) -> pd.DataFrame:
    """Collect fundamental data for multiple symbols.

    Parameters
    ----------
    symbols : str or list of str
        Comma-separated string or list of ticker symbols.
    save_dir : str, optional
        Directory to save per-symbol CSV files.
    start : str, optional
        Start date filter (e.g., "2018-01-01").
    delay : float
        Delay between Yahoo API requests.

    Returns
    -------
    pd.DataFrame
        All symbols' fundamental data concatenated.
    """
    if isinstance(symbols, str):
        symbols = [s.strip().upper() for s in symbols.split(",")]
    else:
        symbols = [s.strip().upper() for s in symbols]

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    all_data = []
    for i, symbol in enumerate(symbols):
        try:
            df = _collect_single_symbol(symbol, start=start)
            if df.empty:
                logger.warning(f"[{i+1}/{len(symbols)}] {symbol}: no data")
                continue
            all_data.append(df)
            logger.info(f"[{i+1}/{len(symbols)}] {symbol}: {len(df)} quarters")

            if save_dir:
                df.to_csv(save_dir / f"{symbol}.csv", index=False)
        except Exception as e:
            logger.warning(f"[{i+1}/{len(symbols)}] {symbol}: error - {e}")

        time.sleep(delay)

    if not all_data:
        return pd.DataFrame()

    result = pd.concat(all_data, ignore_index=True)
    if save_dir:
        result.to_csv(save_dir / "_all_fundamentals.csv", index=False)
        logger.info(f"Saved {len(result)} total records to {save_dir}")

    return result


def collect_from_file(
    symbol_file: str,
    save_dir: str = "./yahoo_fundamental",
    start: Optional[str] = None,
    delay: float = 0.5,
) -> pd.DataFrame:
    """Collect fundamental data from a file containing one symbol per line."""
    symbols = Path(symbol_file).read_text().strip().split("\n")
    symbols = [s.strip() for s in symbols if s.strip()]
    return collect_fundamental_data(symbols, save_dir=save_dir, start=start, delay=delay)


if __name__ == "__main__":
    fire.Fire({"collect": collect_fundamental_data, "collect_from_file": collect_from_file})
