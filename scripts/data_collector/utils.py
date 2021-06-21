#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import re
import time
import bisect
import pickle
import random
import requests
import functools
from pathlib import Path
from typing import Iterable, Tuple, List

import numpy as np
import pandas as pd
from lxml import etree
from loguru import logger
from yahooquery import Ticker
from tqdm import tqdm
from functools import partial
from concurrent.futures import ProcessPoolExecutor

HS_SYMBOLS_URL = "http://app.finance.ifeng.com/hq/list.php?type=stock_a&class={s_type}"

CALENDAR_URL_BASE = "http://push2his.eastmoney.com/api/qt/stock/kline/get?secid={market}.{bench_code}&fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58&klt=101&fqt=0&beg=19900101&end=20991231"
SZSE_CALENDAR_URL = "http://www.szse.cn/api/report/exchange/onepersistenthour/monthList?month={month}&random={random}"

CALENDAR_BENCH_URL_MAP = {
    "CSI300": CALENDAR_URL_BASE.format(market=1, bench_code="000300"),
    "CSI100": CALENDAR_URL_BASE.format(market=1, bench_code="000903"),
    # NOTE: Use the time series of SH600000 as the sequence of all stocks
    "ALL": CALENDAR_URL_BASE.format(market=1, bench_code="000905"),
    # NOTE: Use the time series of ^GSPC(SP500) as the sequence of all stocks
    "US_ALL": "^GSPC",
}


_BENCH_CALENDAR_LIST = None
_ALL_CALENDAR_LIST = None
_HS_SYMBOLS = None
_US_SYMBOLS = None
_EN_FUND_SYMBOLS = None
_CALENDAR_MAP = {}

# NOTE: Until 2020-10-20 20:00:00
MINIMUM_SYMBOLS_NUM = 3900


def get_calendar_list(bench_code="CSI300") -> List[pd.Timestamp]:
    """get SH/SZ history calendar list

    Parameters
    ----------
    bench_code: str
        value from ["CSI300", "CSI500", "ALL", "US_ALL"]

    Returns
    -------
        history calendar list
    """

    logger.info(f"get calendar list: {bench_code}......")

    def _get_calendar(url):
        _value_list = requests.get(url).json()["data"]["klines"]
        return sorted(map(lambda x: pd.Timestamp(x.split(",")[0]), _value_list))

    calendar = _CALENDAR_MAP.get(bench_code, None)
    if calendar is None:
        if bench_code.startswith("US_"):
            df = Ticker(CALENDAR_BENCH_URL_MAP[bench_code]).history(interval="1d", period="max")
            calendar = df.index.get_level_values(level="date").map(pd.Timestamp).unique().tolist()
        else:
            if bench_code.upper() == "ALL":

                @deco_retry
                def _get_calendar(month):
                    _cal = []
                    try:
                        resp = requests.get(SZSE_CALENDAR_URL.format(month=month, random=random.random)).json()
                        for _r in resp["data"]:
                            if int(_r["jybz"]):
                                _cal.append(pd.Timestamp(_r["jyrq"]))
                    except Exception as e:
                        raise ValueError(f"{month}-->{e}")
                    return _cal

                month_range = pd.date_range(start="2000-01", end=pd.Timestamp.now() + pd.Timedelta(days=31), freq="M")
                calendar = []
                for _m in month_range:
                    cal = _get_calendar(_m.strftime("%Y-%m"))
                    if cal:
                        calendar += cal
                calendar = list(filter(lambda x: x <= pd.Timestamp.now(), calendar))
            else:
                calendar = _get_calendar(CALENDAR_BENCH_URL_MAP[bench_code])
        _CALENDAR_MAP[bench_code] = calendar
    logger.info(f"end of get calendar list: {bench_code}.")
    return calendar


def return_date_list(date_field_name: str, file_path: Path):
    date_list = pd.read_csv(file_path, sep=",", index_col=0)[date_field_name].to_list()
    return sorted(map(lambda x: pd.Timestamp(x), date_list))


def get_calendar_list_by_ratio(
    source_dir: [str, Path],
    date_field_name: str = "date",
    threshold: float = 0.5,
    minimum_count: int = 10,
    max_workers: int = 16,
) -> list:
    """get calendar list by selecting the date when few funds trade in this day

    Parameters
    ----------
    source_dir: str or Path
        The directory where the raw data collected from the Internet is saved
    date_field_name: str
            date field name, default is date
    threshold: float
        threshold to exclude some days when few funds trade in this day, default 0.5
    minimum_count: int
        minimum count of funds should trade in one day
    max_workers: int
        Concurrent number, default is 16

    Returns
    -------
        history calendar list
    """
    logger.info(f"get calendar list from {source_dir} by threshold = {threshold}......")

    source_dir = Path(source_dir).expanduser()
    file_list = list(source_dir.glob("*.csv"))

    _number_all_funds = len(file_list)

    logger.info(f"count how many funds trade in this day......")
    _dict_count_trade = dict()  # dict{date:count}
    _fun = partial(return_date_list, date_field_name)
    all_oldest_list = []
    with tqdm(total=_number_all_funds) as p_bar:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for date_list in executor.map(_fun, file_list):
                if date_list:
                    all_oldest_list.append(date_list[0])
                for date in date_list:
                    if date not in _dict_count_trade.keys():
                        _dict_count_trade[date] = 0

                    _dict_count_trade[date] += 1

                p_bar.update()

    logger.info(f"count how many funds have founded in this day......")
    _dict_count_founding = {date: _number_all_funds for date in _dict_count_trade.keys()}  # dict{date:count}
    with tqdm(total=_number_all_funds) as p_bar:
        for oldest_date in all_oldest_list:
            for date in _dict_count_founding.keys():
                if date < oldest_date:
                    _dict_count_founding[date] -= 1

    calendar = [
        date
        for date in _dict_count_trade
        if _dict_count_trade[date] >= max(int(_dict_count_founding[date] * threshold), minimum_count)
    ]

    return calendar


def get_hs_stock_symbols() -> list:
    """get SH/SZ stock symbols

    Returns
    -------
        stock symbols
    """
    global _HS_SYMBOLS

    def _get_symbol():
        _res = set()
        for _k, _v in (("ha", "ss"), ("sa", "sz"), ("gem", "sz")):
            resp = requests.get(HS_SYMBOLS_URL.format(s_type=_k))
            _res |= set(
                map(
                    lambda x: "{}.{}".format(re.findall(r"\d+", x)[0], _v),
                    etree.HTML(resp.text).xpath("//div[@class='result']/ul//li/a/text()"),
                )
            )
            time.sleep(3)
        return _res

    if _HS_SYMBOLS is None:
        symbols = set()
        _retry = 60
        # It may take multiple times to get the complete
        while len(symbols) < MINIMUM_SYMBOLS_NUM:
            symbols |= _get_symbol()
            time.sleep(3)

        symbol_cache_path = Path("~/.cache/hs_symbols_cache.pkl").expanduser().resolve()
        symbol_cache_path.parent.mkdir(parents=True, exist_ok=True)
        if symbol_cache_path.exists():
            with symbol_cache_path.open("rb") as fp:
                cache_symbols = pickle.load(fp)
                symbols |= cache_symbols
        with symbol_cache_path.open("wb") as fp:
            pickle.dump(symbols, fp)

        _HS_SYMBOLS = sorted(list(symbols))

    return _HS_SYMBOLS


def get_us_stock_symbols(qlib_data_path: [str, Path] = None) -> list:
    """get US stock symbols

    Returns
    -------
        stock symbols
    """
    global _US_SYMBOLS

    @deco_retry
    def _get_eastmoney():
        url = "http://4.push2.eastmoney.com/api/qt/clist/get?pn=1&pz=10000&fs=m:105,m:106,m:107&fields=f12"
        resp = requests.get(url)
        if resp.status_code != 200:
            raise ValueError("request error")
        try:
            _symbols = [_v["f12"].replace("_", "-P") for _v in resp.json()["data"]["diff"].values()]
        except Exception as e:
            logger.warning(f"request error: {e}")
            raise
        if len(_symbols) < 8000:
            raise ValueError("request error")
        return _symbols

    @deco_retry
    def _get_nasdaq():
        _res_symbols = []
        for _name in ["otherlisted", "nasdaqtraded"]:
            url = f"ftp://ftp.nasdaqtrader.com/SymbolDirectory/{_name}.txt"
            df = pd.read_csv(url, sep="|")
            df = df.rename(columns={"ACT Symbol": "Symbol"})
            _symbols = df["Symbol"].dropna()
            _symbols = _symbols.str.replace("$", "-P", regex=False)
            _symbols = _symbols.str.replace(".W", "-WT", regex=False)
            _symbols = _symbols.str.replace(".U", "-UN", regex=False)
            _symbols = _symbols.str.replace(".R", "-RI", regex=False)
            _symbols = _symbols.str.replace(".", "-", regex=False)
            _res_symbols += _symbols.unique().tolist()
        return _res_symbols

    @deco_retry
    def _get_nyse():
        url = "https://www.nyse.com/api/quotes/filter"
        _parms = {
            "instrumentType": "EQUITY",
            "pageNumber": 1,
            "sortColumn": "NORMALIZED_TICKER",
            "sortOrder": "ASC",
            "maxResultsPerPage": 10000,
            "filterToken": "",
        }
        resp = requests.post(url, json=_parms)
        if resp.status_code != 200:
            raise ValueError("request error")
        try:
            _symbols = [_v["symbolTicker"].replace("-", "-P") for _v in resp.json()]
        except Exception as e:
            logger.warning(f"request error: {e}")
            _symbols = []
        return _symbols

    if _US_SYMBOLS is None:
        _all_symbols = _get_eastmoney() + _get_nasdaq() + _get_nyse()
        if qlib_data_path is not None:
            for _index in ["nasdaq100", "sp500"]:
                ins_df = pd.read_csv(
                    Path(qlib_data_path).joinpath(f"instruments/{_index}.txt"),
                    sep="\t",
                    names=["symbol", "start_date", "end_date"],
                )
                _all_symbols += ins_df["symbol"].unique().tolist()

        def _format(s_):
            s_ = s_.replace(".", "-")
            s_ = s_.strip("$")
            s_ = s_.strip("*")
            return s_

        _US_SYMBOLS = sorted(set(map(_format, filter(lambda x: len(x) < 8 and not x.endswith("WS"), _all_symbols))))

    return _US_SYMBOLS


def get_en_fund_symbols(qlib_data_path: [str, Path] = None) -> list:
    """get en fund symbols

    Returns
    -------
        fund symbols in China
    """
    global _EN_FUND_SYMBOLS

    @deco_retry
    def _get_eastmoney():
        url = "http://fund.eastmoney.com/js/fundcode_search.js"
        resp = requests.get(url)
        if resp.status_code != 200:
            raise ValueError("request error")
        try:
            _symbols = []
            for sub_data in re.findall(r"[\[](.*?)[\]]", resp.content.decode().split("= [")[-1].replace("];", "")):
                data = sub_data.replace('"', "").replace("'", "")
                # TODO: do we need other informations, like fund_name from ['000001', 'HXCZHH', '华夏成长混合', '混合型', 'HUAXIACHENGZHANGHUNHE']
                _symbols.append(data.split(",")[0])
        except Exception as e:
            logger.warning(f"request error: {e}")
            raise
        if len(_symbols) < 8000:
            raise ValueError("request error")
        return _symbols

    if _EN_FUND_SYMBOLS is None:
        _all_symbols = _get_eastmoney()

        _EN_FUND_SYMBOLS = sorted(set(_all_symbols))

    return _EN_FUND_SYMBOLS


def symbol_suffix_to_prefix(symbol: str, capital: bool = True) -> str:
    """symbol suffix to prefix

    Parameters
    ----------
    symbol: str
        symbol
    capital : bool
        by default True
    Returns
    -------

    """
    code, exchange = symbol.split(".")
    if exchange.lower() in ["sh", "ss"]:
        res = f"sh{code}"
    else:
        res = f"{exchange}{code}"
    return res.upper() if capital else res.lower()


def symbol_prefix_to_sufix(symbol: str, capital: bool = True) -> str:
    """symbol prefix to sufix

    Parameters
    ----------
    symbol: str
        symbol
    capital : bool
        by default True
    Returns
    -------

    """
    res = f"{symbol[:-2]}.{symbol[-2:]}"
    return res.upper() if capital else res.lower()


def deco_retry(retry: int = 5, retry_sleep: int = 3):
    def deco_func(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            _retry = 5 if callable(retry) else retry
            _result = None
            for _i in range(1, _retry + 1):
                try:
                    _result = func(*args, **kwargs)
                    break
                except Exception as e:
                    logger.warning(f"{func.__name__}: {_i} :{e}")
                    if _i == _retry:
                        raise
                time.sleep(retry_sleep)
            return _result

        return wrapper

    return deco_func(retry) if callable(retry) else deco_func


def get_trading_date_by_shift(trading_list: list, trading_date: pd.Timestamp, shift: int = 1):
    """get trading date by shift

    Parameters
    ----------
    trading_list: list
        trading calendar list
    shift : int
        shift, default is 1

    trading_date : pd.Timestamp
        trading date
    Returns
    -------

    """
    trading_date = pd.Timestamp(trading_date)
    left_index = bisect.bisect_left(trading_list, trading_date)
    try:
        res = trading_list[left_index + shift]
    except IndexError:
        res = trading_date
    return res


def generate_minutes_calendar_from_daily(
    calendars: Iterable,
    freq: str = "1min",
    am_range: Tuple[str, str] = ("09:30:00", "11:29:00"),
    pm_range: Tuple[str, str] = ("13:00:00", "14:59:00"),
) -> pd.Index:
    """generate minutes calendar

    Parameters
    ----------
    calendars: Iterable
        daily calendar
    freq: str
        by default 1min
    am_range: Tuple[str, str]
        AM Time Range, by default China-Stock: ("09:30:00", "11:29:00")
    pm_range: Tuple[str, str]
        PM Time Range, by default China-Stock: ("13:00:00", "14:59:00")

    """
    daily_format: str = "%Y-%m-%d"
    res = []
    for _day in calendars:
        for _range in [am_range, pm_range]:
            res.append(
                pd.date_range(
                    f"{pd.Timestamp(_day).strftime(daily_format)} {_range[0]}",
                    f"{pd.Timestamp(_day).strftime(daily_format)} {_range[1]}",
                    freq=freq,
                )
            )

    return pd.Index(sorted(set(np.hstack(res))))


if __name__ == "__main__":
    assert len(get_hs_stock_symbols()) >= MINIMUM_SYMBOLS_NUM
