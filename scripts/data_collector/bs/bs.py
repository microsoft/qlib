import pandas as pd
from scripts.data_collector.base import BaseCollector, BaseNormalize, BaseRun, Normalize
from pathlib import Path
import baostock as bs
from loguru import logger



class BsCollector(BaseCollector):
    def __init__(
        self,
        save_dir: [str, Path],
        start=None,
        end=None,
        interval="1d",
        max_workers=4,
        max_collector_count=2,
        delay=0,
        check_data_length: int = None,
        limit_nums: int = None):

        super(BsCollector, self).__init__(
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
        self.init_datetime()

    def init_datetime(self):
        if self.interval == self.INTERVAL_1min:
            self.start_datetime = max(self.start_datetime, self.DEFAULT_START_DATETIME_1MIN)
        elif self.interval == self.INTERVAL_1d:
            pass
        else:
            raise ValueError(f"interval error: {self.interval}")

        self.start_datetime = self.convert_datetime(self.start_datetime, self._timezone)
        self.end_datetime = self.convert_datetime(self.end_datetime, self._timezone)

    @staticmethod
    def get_data_from_remote(symbol, interval, start, end, show_1min_logging: bool = False):
        error_msg = f"{symbol}-{interval}-{start}-{end}"

        def _show_logging_func():
            if interval == BsCollector.INTERVAL_1min and show_1min_logging:
                logger.warning(f"{error_msg}:{_resp}")

        try:
            lg = bs.login()
            _resp = bs.query_history_k_data_plus(symbol,"date,open,high,low,close,volume,adjustflag",start_date=start.strftime("%Y-%m-%d"), end_date=end.strftime("%Y-%m-%d"),frequency="d", adjustflag="3")
            if isinstance(_resp, pd.DataFrame):
                return _resp.reset_index()
            elif isinstance(_resp, dict):
                _temp_data = _resp.get(symbol, {})
                if isinstance(_temp_data, str) or (
                    isinstance(_resp, dict) and _temp_data.get("indicators", {}).get("quote", None) is None
                ):
                    _show_logging_func()
            else:
                _show_logging_func()
        except Exception as e:
            logger.warning(f"{error_msg}:{e}" + "you can Switch to a foreign network or use another data source")

