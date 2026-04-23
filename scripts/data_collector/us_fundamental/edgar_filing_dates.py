# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Fetch SEC EDGAR filing dates for US stocks.

This module fetches the actual filing dates of 10-Q and 10-K reports from
SEC EDGAR, which is essential for avoiding look-ahead bias when using
fundamental data. Yahoo Finance provides financial statement values but
NOT the date they were publicly filed -- only the report period end date.

Without filing dates, you risk using Q1 data (period ending 3/31) on 4/1,
even though the company might not file until 5/15.

Usage:
    python edgar_filing_dates.py fetch \
        --symbols AAPL,MSFT,GOOGL \
        --save_path ./edgar_filing_dates.csv

    python edgar_filing_dates.py fetch_from_file \
        --symbol_file ./symbols.txt \
        --save_path ./edgar_filing_dates.csv
"""

import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import fire
import pandas as pd
import requests
from loguru import logger

# SEC requires a User-Agent header with contact info
SEC_HEADERS = {
    "User-Agent": "QlibResearch research@example.com",
    "Accept-Encoding": "gzip, deflate",
}

# CIK lookup endpoint
CIK_LOOKUP_URL = "https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&dateRange=custom&forms=10-K,10-Q"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"


def _load_ticker_to_cik_map() -> Dict[str, str]:
    """Load the SEC ticker-to-CIK mapping.

    Returns a dict mapping uppercase ticker symbols to zero-padded CIK strings.
    """
    resp = requests.get(TICKERS_URL, headers=SEC_HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    mapping = {}
    for entry in data.values():
        ticker = str(entry["ticker"]).upper()
        cik = str(entry["cik_str"]).zfill(10)
        mapping[ticker] = cik
    return mapping


def get_filing_dates_for_cik(cik: str) -> pd.DataFrame:
    """Fetch 10-Q and 10-K filing dates from SEC EDGAR for a given CIK.

    Parameters
    ----------
    cik : str
        The CIK number, zero-padded to 10 digits.

    Returns
    -------
    pd.DataFrame
        Columns: [form, filingDate, reportDate]
        - form: "10-Q" or "10-K"
        - filingDate: the date the filing was submitted to SEC (public date)
        - reportDate: the period end date of the financial report
    """
    url = SUBMISSIONS_URL.format(cik=cik)
    resp = requests.get(url, headers=SEC_HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    recent = data.get("filings", {}).get("recent", {})
    if not recent:
        return pd.DataFrame(columns=["form", "filingDate", "reportDate"])

    forms = recent.get("form", [])
    filing_dates = recent.get("filingDate", [])
    report_dates = recent.get("reportDate", [])

    records = []
    for form, f_date, r_date in zip(forms, filing_dates, report_dates):
        if form in ("10-Q", "10-K", "10-Q/A", "10-K/A"):
            records.append(
                {
                    "form": form.replace("/A", ""),  # treat amendments same as original
                    "filingDate": f_date,
                    "reportDate": r_date,
                }
            )

    df = pd.DataFrame(records)
    if not df.empty:
        # Keep only the earliest filing for each (form, reportDate) pair
        # This handles amendments: the original filing date is what matters
        df = df.sort_values("filingDate").drop_duplicates(
            subset=["form", "reportDate"], keep="first"
        )
    return df


def fetch_filing_dates(
    symbols: Union[str, List[str]],
    save_path: Optional[str] = None,
    delay: float = 0.15,
) -> pd.DataFrame:
    """Fetch filing dates for a list of symbols.

    Parameters
    ----------
    symbols : str or list of str
        Comma-separated string or list of ticker symbols.
    save_path : str, optional
        Path to save the result CSV.
    delay : float
        Delay between SEC API requests (SEC rate limit: 10 req/sec).

    Returns
    -------
    pd.DataFrame
        Columns: [symbol, form, filingDate, reportDate]
    """
    if isinstance(symbols, str):
        symbols = [s.strip().upper() for s in symbols.split(",")]
    else:
        symbols = [s.strip().upper() for s in symbols]

    logger.info(f"Loading SEC ticker-to-CIK mapping...")
    try:
        ticker_cik_map = _load_ticker_to_cik_map()
    except Exception as e:
        logger.error(f"Failed to load ticker-to-CIK mapping: {e}")
        return pd.DataFrame()

    all_records = []
    skipped = []

    for i, symbol in enumerate(symbols):
        cik = ticker_cik_map.get(symbol)
        if cik is None:
            skipped.append(symbol)
            continue

        try:
            df = get_filing_dates_for_cik(cik)
            if not df.empty:
                df["symbol"] = symbol
                all_records.append(df)
                logger.info(f"[{i+1}/{len(symbols)}] {symbol}: {len(df)} filings")
            else:
                logger.warning(f"[{i+1}/{len(symbols)}] {symbol}: no filings found")
        except Exception as e:
            logger.warning(f"[{i+1}/{len(symbols)}] {symbol}: error - {e}")

        time.sleep(delay)

    if skipped:
        logger.warning(f"Skipped {len(skipped)} symbols (CIK not found): {skipped[:20]}...")

    if not all_records:
        logger.warning("No filing date data collected")
        return pd.DataFrame(columns=["symbol", "form", "filingDate", "reportDate"])

    result = pd.concat(all_records, ignore_index=True)
    result = result[["symbol", "form", "filingDate", "reportDate"]]
    result["filingDate"] = pd.to_datetime(result["filingDate"])
    result["reportDate"] = pd.to_datetime(result["reportDate"])

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(save_path, index=False)
        logger.info(f"Saved {len(result)} filing records to {save_path}")

    return result


def fetch_from_file(
    symbol_file: str,
    save_path: str = "./edgar_filing_dates.csv",
    delay: float = 0.15,
) -> pd.DataFrame:
    """Fetch filing dates from a file containing one symbol per line."""
    symbols = Path(symbol_file).read_text().strip().split("\n")
    symbols = [s.strip() for s in symbols if s.strip()]
    return fetch_filing_dates(symbols, save_path=save_path, delay=delay)


if __name__ == "__main__":
    fire.Fire({"fetch": fetch_filing_dates, "fetch_from_file": fetch_from_file})
