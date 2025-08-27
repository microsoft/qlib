# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import re
import abc
import sys
from io import BytesIO
from typing import List, Iterable
from pathlib import Path

import fire
import requests
import pandas as pd
from vnstock import Listing
from tqdm import tqdm
from loguru import logger

CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent.parent))

from data_collector.index import IndexBase
from data_collector.utils import get_calendar_list, get_trading_date_by_shift, deco_retry
from data_collector.utils import get_instruments

REQ_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.101 Safari/537.36 Edg/91.0.864.48"
}


class VNIndex(IndexBase):
    """Vietnamese Stock Index implementation using vnstock library"""
    
    def __init__(self, index_name: str, qlib_dir=None, freq: str = "day", request_retry: int = 5, retry_sleep: int = 3):
        super().__init__(index_name, qlib_dir, freq, request_retry, retry_sleep)
        self.listing = Listing(source='VCI')
        
    @property
    def calendar_list(self) -> List[pd.Timestamp]:
        """get history trading date

        Returns
        -------
            calendar list
        """
        _calendar = getattr(self, "_calendar_list", None)
        if not _calendar:
            # For Vietnamese market, create a simple calendar from 2000 to present
            # In a real implementation, you would get actual trading days
            start_date = pd.Timestamp("2000-01-01")
            end_date = pd.Timestamp.now()
            _calendar = pd.bdate_range(start_date, end_date, freq='B').tolist()
            setattr(self, "_calendar_list", _calendar)
        return _calendar

    @property
    @abc.abstractmethod
    def bench_start_date(self) -> pd.Timestamp:
        """
        Returns
        -------
            index start date
        """
        raise NotImplementedError("rewrite bench_start_date")

    @property
    @abc.abstractmethod
    def index_code(self) -> str:
        """
        Returns
        -------
            index code/group name for Vietnamese market
        """
        raise NotImplementedError("rewrite index_code")

    def format_datetime(self, inst_df: pd.DataFrame) -> pd.DataFrame:
        """formatting the datetime in an instrument

        Parameters
        ----------
        inst_df: pd.DataFrame
            inst_df.columns = [self.SYMBOL_FIELD_NAME, self.START_DATE_FIELD, self.END_DATE_FIELD]

        Returns
        -------

        """
        if self.freq != "day":
            inst_df[self.START_DATE_FIELD] = inst_df[self.START_DATE_FIELD].apply(
                lambda x: (pd.Timestamp(x) + pd.Timedelta(hours=9, minutes=30)).strftime("%Y-%m-%d %H:%M:%S")
            )
            inst_df[self.END_DATE_FIELD] = inst_df[self.END_DATE_FIELD].apply(
                lambda x: (pd.Timestamp(x) + pd.Timedelta(hours=15, minutes=0)).strftime("%Y-%m-%d %H:%M:%S")
            )
        return inst_df

    def get_changes(self) -> pd.DataFrame:
        """get companies changes
        
        For Vietnamese market, we'll return an empty DataFrame as change tracking
        is not easily available through vnstock API

        Returns
        -------
            pd.DataFrame:
                symbol      date        type
                ACB      2019-11-11    add
                VCB      2020-11-10    remove
            dtypes:
                symbol: str
                date: pd.Timestamp
                type: str, value from ["add", "remove"]
        """
        # TODO implement this
        logger.info("Vietnamese market change tracking not available through vnstock API")
        # Return empty DataFrame with correct structure
        return pd.DataFrame(columns=[self.SYMBOL_FIELD_NAME, self.DATE_FIELD_NAME, self.CHANGE_TYPE_FIELD])

    def get_new_companies(self) -> pd.DataFrame:
        """Get current companies in the index

        Returns
        -------
            pd.DataFrame:

                symbol     start_date    end_date
                ACB        2000-01-01    2099-12-31
                VCB        2000-01-01    2099-12-31

            dtypes:
                symbol: str
                start_date: pd.Timestamp
                end_date: pd.Timestamp
        """
        logger.info("Getting new companies from Vietnamese market...")
        
        try:
            # Get symbols for the specific group/index
            symbols_df = self.listing.symbols_by_group(group=self.index_code, to_df=True)
            
            if symbols_df.empty:
                logger.warning(f"No symbols found for group {self.index_code}")
                return pd.DataFrame(columns=[self.SYMBOL_FIELD_NAME, self.START_DATE_FIELD, self.END_DATE_FIELD])
            
            # Create DataFrame with proper structure
            df = pd.DataFrame()
            df[self.SYMBOL_FIELD_NAME] = symbols_df
            df[self.START_DATE_FIELD] = self.bench_start_date
            df[self.END_DATE_FIELD] = self.DEFAULT_END_DATE
            
            logger.info(f"Retrieved {len(df)} companies for {self.index_name}")
            return df
            
        except Exception as e:
            logger.error(f"Error getting companies: {e}")
            return pd.DataFrame(columns=[self.SYMBOL_FIELD_NAME, self.START_DATE_FIELD, self.END_DATE_FIELD])

class VN30Index(VNIndex):
    """VN30 Index - Top 30 stocks by market cap on HOSE"""
    
    @property
    def index_code(self):
        return "VN30"

    @property
    def bench_start_date(self) -> pd.Timestamp:
        return pd.Timestamp("2012-01-01")  # VN30 index started around 2012


class VNIndexIndex(VNIndex):
    """VNINDEX - Main stock index of Ho Chi Minh Stock Exchange"""
    
    @property
    def index_code(self):
        return "HOSE"  # All HOSE listed stocks
        
    @property
    def bench_start_date(self) -> pd.Timestamp:
        return pd.Timestamp("2000-01-01")


class HNX30Index(VNIndex):
    """HNX30 Index - Top 30 stocks by market cap on HNX"""
    
    @property
    def index_code(self):
        return "HNX30"

    @property
    def bench_start_date(self) -> pd.Timestamp:
        return pd.Timestamp("2009-01-01")  # HNX30 index started around 2009


class HNXIndex(VNIndex):
    """HNX Index - Main stock index of Hanoi Stock Exchange"""
    
    @property
    def index_code(self):
        return "HNX"  # All HNX listed stocks

    @property
    def bench_start_date(self) -> pd.Timestamp:
        return pd.Timestamp("2005-01-01")


class UPCOMIndex(VNIndex):
    """UPCOM Index - Unlisted Public Company Market"""
    
    @property
    def index_code(self):
        return "UPCOM"

    @property
    def bench_start_date(self) -> pd.Timestamp:
        return pd.Timestamp("2009-01-01")


class VN100Index(VNIndex):
    """VN100 Index - Top 100 stocks by market cap"""
    
    @property
    def index_code(self):
        return "VN100"

    @property
    def bench_start_date(self) -> pd.Timestamp:
        return pd.Timestamp("2010-01-01")


if __name__ == "__main__":
    fire.Fire(get_instruments)
