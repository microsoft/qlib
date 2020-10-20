#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import re
import time
import pickle
import requests
from pathlib import Path

import pandas as pd
from lxml import etree

SYMBOLS_URL = "http://app.finance.ifeng.com/hq/list.php?type=stock_a&class={s_type}"
CSI300_BENCH_URL = "http://push2his.eastmoney.com/api/qt/stock/kline/get?secid=1.000300&fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58&klt=101&fqt=0&beg=19900101&end=20220101"
SH600000_BENCH_URL = "http://push2his.eastmoney.com/api/qt/stock/kline/get?secid=1.600000&fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58&klt=101&fqt=0&beg=19900101&end=20220101"

CALENDAR_URL_BASE = "http://push2his.eastmoney.com/api/qt/stock/kline/get?secid=1.{bench_code}&fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58&klt=101&fqt=0&beg=19900101&end=20220101"

CALENDAR_BENCH_URL_MAP = {
    "CSI300": CALENDAR_URL_BASE.format(bench_code="000300"),
    "CSI100": CALENDAR_URL_BASE.format(bench_code="000903"),
    # NOTE: Use the time series of SH600000 as the sequence of all stocks
    "ALL": CALENDAR_URL_BASE.format(bench_code="600000"),
}

_BENCH_CALENDAR_LIST = None
_ALL_CALENDAR_LIST = None
_HS_SYMBOLS = None
_CALENDAR_MAP = {}

# NOTE: Until 2020-10-20 20:00:00
MINIMUM_SYMBOLS_NUM = 3900


def get_hs_calendar_list(bench_code="CSI300") -> list:
    """get SH/SZ history calendar list

    Parameters
    ----------
    bench_code: str
        value from ["CSI300", "CSI500", "ALL"]

    Returns
    -------
        history calendar list
    """

    def _get_calendar(url):
        _value_list = requests.get(url).json()["data"]["klines"]
        return sorted(map(lambda x: pd.Timestamp(x.split(",")[0]), _value_list))

    calendar = _CALENDAR_MAP.get(bench_code, None)
    if calendar is None:
        calendar = _get_calendar(CALENDAR_BENCH_URL_MAP[bench_code])
        _CALENDAR_MAP[bench_code] = calendar
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
            resp = requests.get(SYMBOLS_URL.format(s_type=_k))
            _res |= set(
                map(
                    lambda x: "{}.{}".format(re.findall(r"\d+", x)[0], _v),
                    etree.HTML(resp.text).xpath("//div[@class='result']/ul//li/a/text()"),
                )
            )
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


if __name__ == '__main__':
    assert len(get_hs_stock_symbols()) >= MINIMUM_SYMBOLS_NUM
