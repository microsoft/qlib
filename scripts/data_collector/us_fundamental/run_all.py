# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
One-command pipeline to collect US fundamental data and prepare it for Qlib.

Usage:
    python run_all.py \
        --symbols AAPL,MSFT,GOOGL,AMZN,META,NVDA \
        --qlib_dir ~/.qlib/qlib_data/us_data \
        --start 2018-01-01 \
        --work_dir ./us_fundamental_workdir

    # Or with a symbol file:
    python run_all.py \
        --symbol_file ./symbols.txt \
        --qlib_dir ~/.qlib/qlib_data/us_data \
        --work_dir ./us_fundamental_workdir

    # Skip SEC EDGAR (faster, slightly less accurate):
    python run_all.py \
        --symbols AAPL,MSFT \
        --qlib_dir ~/.qlib/qlib_data/us_data \
        --skip_edgar \
        --fallback_lag_days 90
"""

import sys
from pathlib import Path
from typing import List, Optional, Union

import fire
from loguru import logger

# Ensure parent directories are importable
CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent.parent))

from us_fundamental.yahoo_fundamental import collect_fundamental_data
from us_fundamental.edgar_filing_dates import fetch_filing_dates
from us_fundamental.build_factors import build


def run(
    qlib_dir: str,
    symbols: Optional[str] = None,
    symbol_file: Optional[str] = None,
    work_dir: str = "./us_fundamental_workdir",
    start: Optional[str] = "2018-01-01",
    skip_edgar: bool = False,
    fallback_lag_days: int = 90,
    yahoo_delay: float = 0.5,
    edgar_delay: float = 0.15,
):
    """Run the complete US fundamental data pipeline.

    Parameters
    ----------
    qlib_dir : str
        Path to existing Qlib data directory with OHLCV data.
    symbols : str, optional
        Comma-separated ticker symbols.
    symbol_file : str, optional
        Path to file with one symbol per line.
    work_dir : str
        Working directory for intermediate files.
    start : str
        Start date for data collection.
    skip_edgar : bool
        Skip SEC EDGAR filing date collection (use fallback lag instead).
    fallback_lag_days : int
        Days to add to reportDate when no EDGAR match. Default 90.
    yahoo_delay : float
        Delay between Yahoo Finance API requests.
    edgar_delay : float
        Delay between SEC EDGAR API requests.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Resolve symbol list
    if symbol_file:
        symbol_list = Path(symbol_file).read_text().strip().split("\n")
        symbol_list = [s.strip() for s in symbol_list if s.strip()]
    elif symbols:
        symbol_list = [s.strip().upper() for s in symbols.split(",")]
    else:
        raise ValueError("Must provide either --symbols or --symbol_file")

    logger.info(f"Pipeline starting for {len(symbol_list)} symbols")

    # ── Step 1: Yahoo Finance ──
    yahoo_dir = work_dir / "yahoo_data"
    logger.info("=" * 60)
    logger.info("Step 1/3: Collecting Yahoo Finance fundamental data...")
    logger.info("=" * 60)
    collect_fundamental_data(
        symbols=symbol_list,
        save_dir=str(yahoo_dir),
        start=start,
        delay=yahoo_delay,
    )
    yahoo_csv = yahoo_dir / "_all_fundamentals.csv"

    # ── Step 2: SEC EDGAR ──
    edgar_csv = work_dir / "edgar_filing_dates.csv"
    if not skip_edgar:
        logger.info("=" * 60)
        logger.info("Step 2/3: Collecting SEC EDGAR filing dates...")
        logger.info("=" * 60)
        fetch_filing_dates(
            symbols=symbol_list,
            save_path=str(edgar_csv),
            delay=edgar_delay,
        )
    else:
        logger.info("=" * 60)
        logger.info(f"Step 2/3: Skipping EDGAR (using {fallback_lag_days}-day lag)")
        logger.info("=" * 60)
        edgar_csv = None

    # ── Step 3: Build factors ──
    output_dir = work_dir / "fundamental_daily"
    logger.info("=" * 60)
    logger.info("Step 3/3: Building daily factor CSVs...")
    logger.info("=" * 60)
    build(
        yahoo_data_path=str(yahoo_csv),
        output_dir=str(output_dir),
        edgar_data_path=str(edgar_csv) if edgar_csv else None,
        qlib_dir=qlib_dir,
        fallback_lag_days=fallback_lag_days,
        start=start,
    )

    # ── Done ──
    logger.info("=" * 60)
    logger.info("Pipeline complete!")
    logger.info(f"Daily factor CSVs: {output_dir}")
    logger.info("")
    logger.info("Next step: dump to Qlib binary format:")
    logger.info(f"  python scripts/dump_bin.py dump_update \\")
    logger.info(f"    --data_path {output_dir} \\")
    logger.info(f"    --qlib_dir {qlib_dir} \\")
    logger.info(f"    --freq day \\")
    logger.info(f'    --exclude_fields symbol,date')
    logger.info("=" * 60)


if __name__ == "__main__":
    fire.Fire(run)
