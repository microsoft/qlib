import re
import requests

import pandas as pd
from lxml import etree

SYMBOLS_URL = "http://app.finance.ifeng.com/hq/list.php?type=stock_a&class={s_type}"
CSI300_BENCH_URL = "http://push2his.eastmoney.com/api/qt/stock/kline/get?secid=1.000300&fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58&klt=101&fqt=0&beg=19900101&end=20220101"
SH600000_BENCH_URL = "http://push2his.eastmoney.com/api/qt/stock/kline/get?secid=1.600000&fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58&klt=101&fqt=0&beg=19900101&end=20220101"

_BENCH_CALENDAR_LIST = None
_ALL_CALENDAR_LIST = None
_HS_SYMBOLS = None


def get_hs_calendar_list(bench=False) -> list:
    """get SH/SZ history calendar list

    Parameters
    ----------
    bench: bool
        whether to get the bench calendar list, by default False

    Returns
    -------
        history calendar list
    """
    global _ALL_CALENDAR_LIST
    global _BENCH_CALENDAR_LIST

    def _get_calendar(url):
        _value_list = requests.get(url).json()["data"]["klines"]
        return sorted(map(lambda x: pd.Timestamp(x.split(",")[0]), _value_list))

    # TODO: get calendar from MSN
    if bench:
        if _BENCH_CALENDAR_LIST is None:
            _BENCH_CALENDAR_LIST = _get_calendar(CSI300_BENCH_URL)
        return _BENCH_CALENDAR_LIST

    if _ALL_CALENDAR_LIST is None:
        _ALL_CALENDAR_LIST = _get_calendar(SH600000_BENCH_URL)
    return _ALL_CALENDAR_LIST


def get_hs_stock_symbols() -> list:
    """get SH/SZ stock symbols

    Returns
    -------
        stock symbols
    """
    global _HS_SYMBOLS
    if _HS_SYMBOLS is None:
        _res = set()
        for _k, _v in (("ha", "ss"), ("sa", "sz"), ("gem", "sz")):
            resp = requests.get(SYMBOLS_URL.format(s_type=_k))
            _res |= set(
                map(
                    lambda x: "{}.{}".format(re.findall(r"\d+", x)[0], _v),
                    etree.HTML(resp.text).xpath("//div[@class='result']/ul//li/a/text()"),
                )
            )
        _HS_SYMBOLS = sorted(list(_res))
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
