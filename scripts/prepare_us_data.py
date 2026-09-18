# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
One-command US stock data preparation for Qlib + Alpha158.

Usage
-----
    # Full pipeline: download from Yahoo, normalize, dump to bin, generate SP500 instruments
    python scripts/prepare_us_data.py all

    # Only download raw CSV from Yahoo Finance
    python scripts/prepare_us_data.py download --start 2000-01-01 --end 2025-01-01

    # Only normalize (requires download first)
    python scripts/prepare_us_data.py normalize

    # Only dump to qlib bin format (requires normalize first)
    python scripts/prepare_us_data.py dump

    # Only generate SP500/NASDAQ100/DJIA instrument lists
    python scripts/prepare_us_data.py instruments

    # Use pre-packaged data from Azure Blob instead of Yahoo
    python scripts/prepare_us_data.py from_qlib_data

    # Full pipeline with custom paths
    python scripts/prepare_us_data.py all --source_dir ~/my_data/source --qlib_dir ~/my_data/us_data
"""

import sys
import datetime
from pathlib import Path

import fire
import pandas as pd
from loguru import logger

CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR))


DEFAULT_QLIB_DIR = "~/.qlib/qlib_data/us_data"
DEFAULT_SOURCE_DIR = "~/.qlib/stock_data/source/us_data"
DEFAULT_NORMALIZE_DIR = "~/.qlib/stock_data/normalize/us_data"


class PrepareUSData:
    """One-command US stock data preparation pipeline."""

    def __init__(
        self,
        source_dir: str = DEFAULT_SOURCE_DIR,
        normalize_dir: str = DEFAULT_NORMALIZE_DIR,
        qlib_dir: str = DEFAULT_QLIB_DIR,
        max_workers: int = 1,
    ):
        self.source_dir = Path(source_dir).expanduser().resolve()
        self.normalize_dir = Path(normalize_dir).expanduser().resolve()
        self.qlib_dir = Path(qlib_dir).expanduser().resolve()
        self.max_workers = max_workers

    # ------------------------------------------------------------------
    # Step 1: Download raw CSV from Yahoo Finance
    # ------------------------------------------------------------------
    def download(
        self,
        start: str = "2000-01-01",
        end: str = None,
        delay: float = 1.0,
        max_collector_count: int = 2,
        check_data_length: int = None,
        limit_nums: int = None,
    ):
        """Download US stock OHLCV data from Yahoo Finance.

        Parameters
        ----------
        start : str
            Start date (inclusive), default "2000-01-01".
        end : str
            End date (exclusive), default today.
        delay : float
            Seconds between API requests, default 1.0.
        """
        if end is None:
            end = pd.Timestamp(datetime.datetime.now()).strftime("%Y-%m-%d")

        logger.info(f"[Step 1/4] Downloading US stock data: {start} ~ {end}")
        logger.info(f"  source_dir: {self.source_dir}")

        from data_collector.yahoo.collector import Run as YahooRun

        runner = YahooRun(
            source_dir=str(self.source_dir),
            normalize_dir=str(self.normalize_dir),
            max_workers=self.max_workers,
            interval="1d",
            region="US",
        )
        runner.download_data(
            max_collector_count=max_collector_count,
            delay=delay,
            start=start,
            end=end,
            check_data_length=check_data_length,
            limit_nums=limit_nums,
        )
        logger.info("[Step 1/4] Download complete.")

    # ------------------------------------------------------------------
    # Step 2: Normalize (adjust price + scale)
    # ------------------------------------------------------------------
    def normalize(self):
        """Normalize downloaded CSV data (adjust price, scale to first close = 1)."""
        logger.info("[Step 2/4] Normalizing data...")
        logger.info(f"  source_dir:    {self.source_dir}")
        logger.info(f"  normalize_dir: {self.normalize_dir}")

        from data_collector.yahoo.collector import Run as YahooRun

        runner = YahooRun(
            source_dir=str(self.source_dir),
            normalize_dir=str(self.normalize_dir),
            max_workers=self.max_workers,
            interval="1d",
            region="US",
        )
        runner.normalize_data(
            date_field_name="date",
            symbol_field_name="symbol",
        )
        logger.info("[Step 2/4] Normalize complete.")

    # ------------------------------------------------------------------
    # Step 3: Dump to qlib binary format
    # ------------------------------------------------------------------
    def dump(self):
        """Convert normalized CSV to qlib binary format."""
        import multiprocessing

        logger.info("[Step 3/4] Dumping to qlib binary format...")
        logger.info(f"  normalize_dir: {self.normalize_dir}")
        logger.info(f"  qlib_dir:      {self.qlib_dir}")

        from dump_bin import DumpDataAll

        dumper = DumpDataAll(
            data_path=str(self.normalize_dir),
            qlib_dir=str(self.qlib_dir),
            freq="day",
            max_workers=max(multiprocessing.cpu_count() - 2, 1),
            exclude_fields="date,symbol",
            file_suffix=".csv",
        )
        dumper.dump()
        logger.info("[Step 3/4] Dump complete.")

    # ------------------------------------------------------------------
    # Step 4: Generate US index instrument lists
    # ------------------------------------------------------------------
    def instruments(self, index_list: str = "SP500,NASDAQ100,DJIA,SP400"):
        """Generate instrument lists for US indices (SP500, NASDAQ100, etc.).

        Parameters
        ----------
        index_list : str
            Comma-separated index names, default "SP500,NASDAQ100,DJIA,SP400".
        """
        logger.info("[Step 4/4] Generating US index instrument files...")
        logger.info(f"  qlib_dir:   {self.qlib_dir}")
        logger.info(f"  indices:    {index_list}")

        sys.path.append(str(CUR_DIR / "data_collector"))

        from data_collector.us_index.collector import get_instruments

        for index_name in index_list.split(","):
            index_name = index_name.strip()
            if not index_name:
                continue
            logger.info(f"  Generating {index_name}...")
            try:
                get_instruments(
                    str(self.qlib_dir),
                    index_name,
                    market_index="us_index",
                )
            except Exception as e:
                logger.warning(f"  Failed to generate {index_name}: {e}")
        logger.info("[Step 4/4] Instruments generation complete.")

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------
    def all(
        self,
        start: str = "2000-01-01",
        end: str = None,
        delay: float = 1.0,
        max_collector_count: int = 2,
        check_data_length: int = None,
        limit_nums: int = None,
        index_list: str = "SP500,NASDAQ100,DJIA,SP400",
    ):
        """Run the full pipeline: download -> normalize -> dump -> instruments.

        Parameters
        ----------
        start : str
            Start date (inclusive), default "2000-01-01".
        end : str
            End date (exclusive), default today.
        delay : float
            Seconds between Yahoo API requests, default 1.0.
        index_list : str
            Comma-separated index names for instrument generation.
        """
        logger.info("=" * 60)
        logger.info("US Stock Data Preparation Pipeline")
        logger.info("=" * 60)
        logger.info(f"  source_dir:    {self.source_dir}")
        logger.info(f"  normalize_dir: {self.normalize_dir}")
        logger.info(f"  qlib_dir:      {self.qlib_dir}")
        logger.info("=" * 60)

        self.download(
            start=start,
            end=end,
            delay=delay,
            max_collector_count=max_collector_count,
            check_data_length=check_data_length,
            limit_nums=limit_nums,
        )
        self.normalize()
        self.dump()
        self.instruments(index_list=index_list)

        logger.info("=" * 60)
        logger.info("All done! You can now use the data:")
        logger.info("")
        logger.info("  import qlib")
        logger.info(f'  qlib.init(provider_uri="{self.qlib_dir}", region="us")')
        logger.info("")
        logger.info("  from qlib.contrib.data.handler import Alpha158")
        logger.info('  h = Alpha158(instruments="sp500", start_time="2008-01-01", end_time="2024-12-31")')
        logger.info("  df = h.fetch()  # shape: (N, 158)")
        logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Alternative: use pre-packaged data from Azure Blob
    # ------------------------------------------------------------------
    def from_qlib_data(self, delete_old: bool = True, exists_skip: bool = False):
        """Download pre-packaged US data from Qlib's Azure Blob storage.

        This is the fastest way to get started, but data may not be up-to-date.
        After downloading, you can use `update_data_to_bin` to update incrementally.
        """
        logger.info("Downloading pre-packaged US data from Azure Blob...")
        logger.info(f"  qlib_dir: {self.qlib_dir}")

        from qlib.tests.data import GetData

        GetData().qlib_data(
            name="qlib_data",
            target_dir=str(self.qlib_dir),
            version=None,
            interval="1d",
            region="us",
            delete_old=delete_old,
            exists_skip=exists_skip,
        )
        logger.info("Download complete. Generating instruments...")
        self.instruments()


if __name__ == "__main__":
    fire.Fire(PrepareUSData)
