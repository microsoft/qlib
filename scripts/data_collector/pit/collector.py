# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import re
import sys
import datetime
from pathlib import Path

import fire
import numpy as np
import pandas as pd
import baostock as bs
from loguru import logger

CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent.parent))
from data_collector.base import BaseCollector, BaseRun
from data_collector.utils import get_calendar_list, get_hs_stock_symbols


class PitCollector(BaseCollector):

    DEFAULT_START_DATETIME_QUARTER = pd.Timestamp("2000-01-01")
    DEFAULT_START_DATETIME_ANNUAL = pd.Timestamp("2000-01-01")
    DEFAULT_END_DATETIME_QUARTER = pd.Timestamp(datetime.datetime.now() + pd.Timedelta(days=1))
    DEFAULT_END_DATETIME_ANNUAL = pd.Timestamp(datetime.datetime.now() + pd.Timedelta(days=1))

    INTERVAL_quarterly = "quarterly"
    INTERVAL_annual = "annual"

    def __init__(
        self,
        save_dir: [str, Path],
        start=None,
        end=None,
        interval="quarterly",
        max_workers=1,
        max_collector_count=1,
        delay=0,
        check_data_length: bool = False,
        limit_nums: int = None,
        symbol_flt_regx=None,
    ):
        """

        Parameters
        ----------
        save_dir: str
            pit save dir
        interval: str:
            value from ['quarterly', 'annual']
        max_workers: int
            workers, default 1
        max_collector_count: int
            default 1
        delay: float
            time.sleep(delay), default 0
        start: str
            start datetime, default None
        end: str
            end datetime, default None
        limit_nums: int
            using for debug, by default None
        """
        if symbol_flt_regx is None:
            self.symbol_flt_regx = None
        else:
            self.symbol_flt_regx = re.compile(symbol_flt_regx)
        super(PitCollector, self).__init__(
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

    def normalize_symbol(self, symbol):
        symbol_s = symbol.split(".")
        symbol = f"sh{symbol_s[0]}" if symbol_s[-1] == "ss" else f"sz{symbol_s[0]}"
        return symbol

    def get_instrument_list(self):
        logger.info("get cn stock symbols......")
        symbols = get_hs_stock_symbols()
        logger.info(f"get {symbols[:10]}[{len(symbols)}] symbols.")
        if self.symbol_flt_regx is not None:
            s_flt = []
            for s in symbols:
                m = self.symbol_flt_regx.match(s)
                if m is not None:
                    s_flt.append(s)
            logger.info(f"after filtering, it becomes {s_flt[:10]}[{len(s_flt)}] symbols")
            return s_flt

        return symbols

    def _get_data_from_baostock(self, symbol, interval, start_datetime, end_datetime):
        error_msg = f"{symbol}-{interval}-{start_datetime}-{end_datetime}"

        def _str_to_float(r):
            try:
                return float(r)
            except Exception as e:
                return np.nan

        try:
            code, market = symbol.split(".")
            market = {"ss": "sh"}.get(market, market)  # baostock's API naming is different from default symbol list
            symbol = f"{market}.{code}"
            rs_report = bs.query_performance_express_report(
                code=symbol,
                start_date=str(start_datetime.date()),
                end_date=str(end_datetime.date()),
            )
            report_list = []
            while (rs_report.error_code == "0") & rs_report.next():
                report_list.append(rs_report.get_row_data())

            df_report = pd.DataFrame(report_list, columns=rs_report.fields)
            if {
                "performanceExpPubDate",
                "performanceExpStatDate",
                "performanceExpressROEWa",
            } <= set(rs_report.fields):
                df_report = df_report[
                    [
                        "performanceExpPubDate",
                        "performanceExpStatDate",
                        "performanceExpressROEWa",
                    ]
                ]
                df_report.rename(
                    columns={
                        "performanceExpPubDate": "date",
                        "performanceExpStatDate": "period",
                        "performanceExpressROEWa": "value",
                    },
                    inplace=True,
                )
                df_report["value"] = df_report["value"].apply(lambda r: _str_to_float(r) / 100.0)
                df_report["field"] = "roeWa"

            profit_list = []
            for year in range(start_datetime.year - 1, end_datetime.year + 1):
                for q_num in range(0, 4):
                    rs_profit = bs.query_profit_data(code=symbol, year=year, quarter=q_num + 1)
                    while (rs_profit.error_code == "0") & rs_profit.next():
                        row_data = rs_profit.get_row_data()
                        if "pubDate" in rs_profit.fields:
                            pub_date = pd.Timestamp(row_data[rs_profit.fields.index("pubDate")])
                            if pub_date >= start_datetime and pub_date <= end_datetime:
                                profit_list.append(row_data)

            df_profit = pd.DataFrame(profit_list, columns=rs_profit.fields)
            if {"pubDate", "statDate", "roeAvg"} <= set(rs_profit.fields):
                df_profit = df_profit[["pubDate", "statDate", "roeAvg"]]
                df_profit.rename(
                    columns={
                        "pubDate": "date",
                        "statDate": "period",
                        "roeAvg": "value",
                    },
                    inplace=True,
                )
                df_profit["value"] = df_profit["value"].apply(_str_to_float)
                df_profit["field"] = "roeWa"

            forecast_list = []
            rs_forecast = bs.query_forecast_report(
                code=symbol,
                start_date=str(start_datetime.date()),
                end_date=str(end_datetime.date()),
            )

            while (rs_forecast.error_code == "0") & rs_forecast.next():
                forecast_list.append(rs_forecast.get_row_data())

            df_forecast = pd.DataFrame(forecast_list, columns=rs_forecast.fields)
            if {
                "profitForcastExpPubDate",
                "profitForcastExpStatDate",
                "profitForcastChgPctUp",
                "profitForcastChgPctDwn",
            } <= set(rs_forecast.fields):
                df_forecast = df_forecast[
                    [
                        "profitForcastExpPubDate",
                        "profitForcastExpStatDate",
                        "profitForcastChgPctUp",
                        "profitForcastChgPctDwn",
                    ]
                ]
                df_forecast.rename(
                    columns={
                        "profitForcastExpPubDate": "date",
                        "profitForcastExpStatDate": "period",
                    },
                    inplace=True,
                )

                df_forecast["profitForcastChgPctUp"] = df_forecast["profitForcastChgPctUp"].apply(_str_to_float)
                df_forecast["profitForcastChgPctDwn"] = df_forecast["profitForcastChgPctDwn"].apply(_str_to_float)
                df_forecast["value"] = (
                    df_forecast["profitForcastChgPctUp"] + df_forecast["profitForcastChgPctDwn"]
                ) / 200
                df_forecast["field"] = "YOYNI"
                df_forecast.drop(
                    ["profitForcastChgPctUp", "profitForcastChgPctDwn"],
                    axis=1,
                    inplace=True,
                )

            growth_list = []
            for year in range(start_datetime.year - 1, end_datetime.year + 1):
                for q_num in range(0, 4):
                    rs_growth = bs.query_growth_data(code=symbol, year=year, quarter=q_num + 1)
                    while (rs_growth.error_code == "0") & rs_growth.next():
                        row_data = rs_growth.get_row_data()
                        if "pubDate" in rs_growth.fields:
                            pub_date = pd.Timestamp(row_data[rs_growth.fields.index("pubDate")])
                            if pub_date >= start_datetime and pub_date <= end_datetime:
                                growth_list.append(row_data)

            df_growth = pd.DataFrame(growth_list, columns=rs_growth.fields)
            if {"pubDate", "statDate", "YOYNI"} <= set(rs_growth.fields):
                df_growth = df_growth[["pubDate", "statDate", "YOYNI"]]
                df_growth.rename(
                    columns={"pubDate": "date", "statDate": "period", "YOYNI": "value"},
                    inplace=True,
                )
                df_growth["value"] = df_growth["value"].apply(_str_to_float)
                df_growth["field"] = "YOYNI"
            df_merge = df_report.append([df_profit, df_forecast, df_growth])

            return df_merge
        except Exception as e:
            logger.warning(f"{error_msg}:{e}")

    def _process_data(self, df, symbol, interval):
        error_msg = f"{symbol}-{interval}"

        def _process_period(r):
            _date = pd.Timestamp(r)
            return _date.year if interval == self.INTERVAL_annual else _date.year * 100 + (_date.month - 1) // 3 + 1

        try:
            _date = df["period"].apply(
                lambda x: (
                    pd.to_datetime(x) + pd.DateOffset(days=(45 if interval == self.INTERVAL_quarterly else 90))
                ).date()
            )
            df["date"] = df["date"].fillna(_date.astype(str))
            df["period"] = df["period"].apply(_process_period)
            return df
        except Exception as e:
            logger.warning(f"{error_msg}:{e}")

    def get_data(
        self,
        symbol: str,
        interval: str,
        start_datetime: pd.Timestamp,
        end_datetime: pd.Timestamp,
    ) -> [pd.DataFrame]:

        if interval == self.INTERVAL_quarterly:
            _result = self._get_data_from_baostock(symbol, interval, start_datetime, end_datetime)
            if _result is None or _result.empty:
                return _result
            else:
                return self._process_data(_result, symbol, interval)
        else:
            raise ValueError(f"cannot support {interval}")
        return self._process_data(_result, interval)

    @property
    def min_numbers_trading(self):
        pass


class Run(BaseRun):
    def __init__(self, source_dir=None, max_workers=1, interval="quarterly"):
        """

        Parameters
        ----------
        source_dir: str
            The directory where the raw data collected from the Internet is saved, default "Path(__file__).parent/source"
        max_workers: int
            Concurrent number, default is 4
        interval: str
            freq, value from [quarterly, annual], default 1d
        """
        super().__init__(source_dir=source_dir, max_workers=max_workers, interval=interval)

    @property
    def collector_class_name(self):
        return "PitCollector"

    @property
    def default_base_dir(self) -> [Path, str]:
        return CUR_DIR

    def download_data(
        self,
        max_collector_count=1,
        delay=0,
        start=None,
        end=None,
        check_data_length=False,
        limit_nums=None,
        **kwargs,
    ):
        """download data from Internet

        Parameters
        ----------
        max_collector_count: int
            default 2
        delay: float
            time.sleep(delay), default 0
        start: str
            start datetime, default "2000-01-01"
        end: str
            end datetime, default ``pd.Timestamp(datetime.datetime.now() + pd.Timedelta(days=1))``
        check_data_length: bool # if this param useful?
            check data length, by default False
        limit_nums: int
            using for debug, by default None

        Examples
        ---------
            # get quarterly data
            $ python collector.py download_data --source_dir ~/.qlib/cn_data/source/pit_quarter --start 2000-01-01 --end 2021-01-01 --interval quarterly
        """

        super(Run, self).download_data(
            max_collector_count,
            delay,
            start,
            end,
            check_data_length,
            limit_nums,
            **kwargs,
        )

    def normalize_class_name(self):
        pass


if __name__ == "__main__":
    bs.login()
    fire.Fire(Run)
    bs.logout()
