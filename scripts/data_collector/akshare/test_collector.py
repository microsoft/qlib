"""Unit tests for the AKShare collector.

Run from the repo root:

    python -m pytest scripts/data_collector/akshare/test_collector.py -v
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

CUR_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CUR_DIR.parent.parent))

from data_collector.akshare import collector as akshare_collector
from data_collector.akshare.collector import (
    AKShareCollector,
    AKShareNormalize,
    get_all_symbols,
    qlib_to_raw,
    symbol_to_qlib,
)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("600519", "SH600519"),
        ("000001", "SZ000001"),
        ("300750", "SZ300750"),
        ("sh600519", "SH600519"),
        ("sz000001", "SZ000001"),
        ("SH600519", "SH600519"),
        (" 000001 ", "SZ000001"),
    ],
)
def test_symbol_to_qlib(raw, expected):
    assert symbol_to_qlib(raw) == expected


@pytest.mark.parametrize(
    "qlib_symbol, expected",
    [
        ("SH600519", "600519"),
        ("SZ000001", "000001"),
        ("sh600519", "600519"),
        ("600519", "600519"),
    ],
)
def test_qlib_to_raw(qlib_symbol, expected):
    assert qlib_to_raw(qlib_symbol) == expected


def _make_collector(tmp_path, **kwargs):
    defaults = dict(
        save_dir=tmp_path,
        start="2024-01-01",
        end="2024-01-10",
        symbols="600519,000001",
    )
    defaults.update(kwargs)
    return AKShareCollector(**defaults)


def test_parse_symbols_string(tmp_path):
    c = _make_collector(tmp_path, symbols="600519, 000001 ,000001")
    assert c.requested_symbols == ["000001", "600519"]


def test_parse_symbols_list(tmp_path):
    c = _make_collector(tmp_path, symbols=["600519", "000001"])
    assert c.requested_symbols == ["000001", "600519"]


def test_parse_symbols_from_file(tmp_path):
    sym_file = tmp_path / "symbols.txt"
    sym_file.write_text("600519\n000001\n\n300750\n", encoding="utf-8")
    c = _make_collector(tmp_path, symbols=None, symbol_file=str(sym_file))
    assert c.requested_symbols == ["000001", "300750", "600519"]


def test_parse_symbols_combined(tmp_path):
    sym_file = tmp_path / "symbols.txt"
    sym_file.write_text("600519\n", encoding="utf-8")
    c = _make_collector(tmp_path, symbols="000001", symbol_file=str(sym_file))
    assert c.requested_symbols == ["000001", "600519"]


def test_parse_symbols_no_args_triggers_discover(tmp_path, monkeypatch):
    """When no --symbols or --symbol_file given, _parse_symbols auto-discovers."""
    fake_df = pd.DataFrame({"code": ["600519", "000001"], "name": ["a", "b"]})
    monkeypatch.setattr(akshare_collector.ak, "stock_info_a_code_name", lambda: fake_df)
    c = AKShareCollector(save_dir=tmp_path, start="2024-01-01", end="2024-01-10")
    assert c.requested_symbols == ["000001", "600519"]


def test_normalize_symbol_routes_to_helper(tmp_path):
    c = _make_collector(tmp_path)
    assert c.normalize_symbol("600519") == "SH600519"
    assert c.normalize_symbol("000001") == "SZ000001"


def test_get_instrument_list_returns_requested(tmp_path):
    c = _make_collector(tmp_path, symbols="600519,000001")
    assert c.get_instrument_list() == ["000001", "600519"]


def _fake_akshare_df():
    return pd.DataFrame(
        {
            "日期": ["2024-01-02", "2024-01-03"],
            "股票代码": ["600519", "600519"],
            "开盘": ["1700.0", "1710.0"],
            "收盘": ["1705.0", "1715.0"],
            "最高": ["1720.0", "1725.0"],
            "最低": ["1690.0", "1700.0"],
            "成交量": ["100000", "120000"],
            "成交额": ["170500000", "172500000"],
            "振幅": ["1.0", "1.1"],
        }
    )


def test_get_data_success(tmp_path, monkeypatch):
    captured = {}

    def fake_hist(symbol, period, start_date, end_date, adjust):
        captured.update(
            symbol=symbol,
            period=period,
            start_date=start_date,
            end_date=end_date,
            adjust=adjust,
        )
        return _fake_akshare_df()

    monkeypatch.setattr(akshare_collector.ak, "stock_zh_a_hist", fake_hist)

    c = _make_collector(tmp_path, symbols="600519")
    df = c.get_data(
        "SH600519",
        interval="1d",
        start_datetime=pd.Timestamp("2024-01-01"),
        end_datetime=pd.Timestamp("2024-01-10"),
    )

    assert captured == {
        "symbol": "600519",
        "period": "daily",
        "start_date": "20240101",
        "end_date": "20240110",
        "adjust": "qfq",
    }
    assert list(df.columns) == ["date", "symbol", "open", "close", "high", "low", "volume", "money"]
    assert df["symbol"].unique().tolist() == ["SH600519"]
    assert df["open"].dtype.kind == "f"
    assert df["volume"].dtype.kind in ("f", "i")
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert len(df) == 2


def test_get_data_handles_empty_response(tmp_path, monkeypatch):
    monkeypatch.setattr(
        akshare_collector.ak, "stock_zh_a_hist", lambda **_: pd.DataFrame()
    )
    c = _make_collector(tmp_path, symbols="600519")
    df = c.get_data(
        "SH600519",
        interval="1d",
        start_datetime=pd.Timestamp("2024-01-01"),
        end_datetime=pd.Timestamp("2024-01-10"),
    )
    assert df.empty


def test_get_data_swallows_akshare_exception(tmp_path, monkeypatch):
    def boom(**_):
        raise RuntimeError("upstream HTTP 500")

    monkeypatch.setattr(akshare_collector.ak, "stock_zh_a_hist", boom)
    c = _make_collector(tmp_path, symbols="600519")
    df = c.get_data(
        "SH600519",
        interval="1d",
        start_datetime=pd.Timestamp("2024-01-01"),
        end_datetime=pd.Timestamp("2024-01-10"),
    )
    assert df.empty


def test_get_data_passes_through_adjust(tmp_path, monkeypatch):
    captured = {}

    def fake_hist(**kwargs):
        captured.update(kwargs)
        return _fake_akshare_df()

    monkeypatch.setattr(akshare_collector.ak, "stock_zh_a_hist", fake_hist)
    c = _make_collector(tmp_path, symbols="600519", adjust="hfq")
    c.get_data(
        "SH600519",
        interval="1d",
        start_datetime=pd.Timestamp("2024-01-01"),
        end_datetime=pd.Timestamp("2024-01-10"),
    )
    assert captured["adjust"] == "hfq"


class _NormalizeNoCalendar(AKShareNormalize):
    def __init__(self):
        self._date_field_name = "date"
        self._symbol_field_name = "symbol"
        self._calendar_list = []


def test_normalize_dedups_and_sorts():
    raw = pd.DataFrame(
        {
            "date": ["2024-01-03", "2024-01-02", "2024-01-02"],
            "symbol": ["SH600519"] * 3,
            "close": [1715.0, 1705.0, 1705.0],
        }
    )
    out = _NormalizeNoCalendar().normalize(raw)
    assert len(out) == 2
    assert out["date"].is_monotonic_increasing
    assert pd.api.types.is_datetime64_any_dtype(out["date"])


def test_normalize_passthrough_empty():
    out = _NormalizeNoCalendar().normalize(pd.DataFrame())
    assert out.empty


# --- get_all_symbols tests ---


def test_get_all_symbols_filters_bshares(monkeypatch):
    fake_df = pd.DataFrame(
        {
            "code": ["600519", "000001", "300750", "900901", "200002", "688001"],
            "name": ["贵州茅台", "平安银行", "宁德时代", "dummy", "dummy", "dummy"],
        }
    )
    monkeypatch.setattr(akshare_collector.ak, "stock_info_a_code_name", lambda: fake_df)
    result = get_all_symbols()
    assert "600519" in result
    assert "000001" in result
    assert "300750" in result
    assert "688001" in result  # STAR board included
    assert "900901" not in result  # B-share excluded
    assert "200002" not in result  # B-share excluded


def test_get_all_symbols_deduplicates(monkeypatch):
    fake_df = pd.DataFrame({"code": ["600519", "600519", "000001"], "name": ["a", "b", "c"]})
    monkeypatch.setattr(akshare_collector.ak, "stock_info_a_code_name", lambda: fake_df)
    result = get_all_symbols()
    assert result == ["000001", "600519"]


# --- _parse_symbols auto-discover fallback ---


def test_parse_symbols_auto_discover(tmp_path, monkeypatch):
    fake_df = pd.DataFrame(
        {"code": ["600519", "000001"], "name": ["贵州茅台", "平安银行"]}
    )
    monkeypatch.setattr(akshare_collector.ak, "stock_info_a_code_name", lambda: fake_df)
    c = AKShareCollector(save_dir=tmp_path, start="2024-01-01", end="2024-01-10")
    assert "000001" in c.requested_symbols
    assert "600519" in c.requested_symbols


# --- _get_calendar_list test ---


def test_get_calendar_list_uses_shared_utils(monkeypatch):
    call_args = {}

    def fake_get_calendar_list(bench_code):
        call_args["bench_code"] = bench_code
        return [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]

    monkeypatch.setattr(
        "data_collector.akshare.collector.get_calendar_list", fake_get_calendar_list
    )
    norm = AKShareNormalize()
    assert call_args["bench_code"] == "ALL"
    assert len(norm._calendar_list) == 2
