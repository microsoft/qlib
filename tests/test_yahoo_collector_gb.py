# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unit tests for the GB (London Stock Exchange) Yahoo Finance data collector.

Covers:
- get_gb_stock_symbols: pagination, .L filtering, caching behaviour
- get_calendar_list routing for "GB_ALL" -> ^FTSE
- YahooCollectorGB: timezone, normalize_symbol, instrument list
- YahooNormalizeGB1d / YahooNormalizeGB1min class instantiation
- Run class-name resolution for GB region
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pandas as pd

# ---------------------------------------------------------------------------
# Ensure the data_collector package is on the path (mirrors collector.py setup)
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
_COLLECTOR_DIR = _SCRIPTS_DIR / "data_collector"
_YAHOO_DIR = _COLLECTOR_DIR / "yahoo"

for _p in [str(_SCRIPTS_DIR), str(_COLLECTOR_DIR), str(_YAHOO_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import data_collector.utils as dc_utils  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_screener_page(symbols: list) -> MagicMock:
    """Return a mock requests.Response for one screener page."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "finance": {
            "result": [{"quotes": [{"symbol": s} for s in symbols]}]
        }
    }
    return mock_resp


# ---------------------------------------------------------------------------
# Tests: get_gb_stock_symbols
# ---------------------------------------------------------------------------


class TestGetGBStockSymbols(unittest.TestCase):
    def setUp(self):
        # Reset the module-level cache before every test
        dc_utils._GB_SYMBOLS = None

    def test_filters_dot_l_symbols_only(self):
        """Only symbols ending in '.L' should be retained."""
        mock_resp = _make_screener_page(["HSBA.L", "AZN.L", "AAPL", "7203.T", "BP.L"])
        with patch("data_collector.utils.requests.get", return_value=mock_resp):
            symbols = dc_utils.get_gb_stock_symbols()
        self.assertIn("HSBA.L", symbols)
        self.assertIn("AZN.L", symbols)
        self.assertIn("BP.L", symbols)
        self.assertNotIn("AAPL", symbols)
        self.assertNotIn("7203.T", symbols)

    def test_pagination_stops_on_empty_page(self):
        """Pagination must stop when the API returns an empty quotes list."""
        mock_full = _make_screener_page(["HSBA.L", "AZN.L"])
        mock_empty = _make_screener_page([])
        with patch("data_collector.utils.requests.get", side_effect=[mock_full, mock_empty]):
            symbols = dc_utils.get_gb_stock_symbols()
        self.assertEqual(symbols, sorted({"HSBA.L", "AZN.L"}))

    def test_pagination_stops_when_page_smaller_than_page_size(self):
        """Pagination must stop when len(quotes) < 250 without a second request."""
        mock_partial = _make_screener_page(["HSBA.L", "AZN.L", "SHEL.L"])
        with patch("data_collector.utils.requests.get", return_value=mock_partial) as mock_get:
            dc_utils.get_gb_stock_symbols()
        self.assertEqual(mock_get.call_count, 1)

    def test_result_is_sorted(self):
        """Returned list must be in sorted order."""
        mock_resp = _make_screener_page(["SHEL.L", "AZN.L", "BP.L"])
        with patch("data_collector.utils.requests.get", return_value=mock_resp):
            symbols = dc_utils.get_gb_stock_symbols()
        self.assertEqual(symbols, sorted(symbols))

    def test_result_is_deduplicated(self):
        """Duplicate symbols across pages must appear only once."""
        page1 = _make_screener_page(["HSBA.L", "AZN.L"])
        # page2 has a duplicate from page1 — still < 250 so stops after page2
        page2 = _make_screener_page(["AZN.L", "BP.L"])
        with patch("data_collector.utils.requests.get", side_effect=[page1, page2]):
            symbols = dc_utils.get_gb_stock_symbols()
        self.assertEqual(symbols.count("AZN.L"), 1)

    def test_cache_is_populated_after_first_call(self):
        """_GB_SYMBOLS must be set after the first call."""
        mock_resp = _make_screener_page(["BP.L"])
        with patch("data_collector.utils.requests.get", return_value=mock_resp):
            dc_utils.get_gb_stock_symbols()
        self.assertIsNotNone(dc_utils._GB_SYMBOLS)

    def test_cache_prevents_second_http_request(self):
        """A second call must not make another HTTP request."""
        mock_resp = _make_screener_page(["BP.L"])
        with patch("data_collector.utils.requests.get", return_value=mock_resp) as mock_get:
            dc_utils.get_gb_stock_symbols()
            dc_utils.get_gb_stock_symbols()
        self.assertEqual(mock_get.call_count, 1)


# ---------------------------------------------------------------------------
# Tests: get_calendar_list routing for GB_ALL
# ---------------------------------------------------------------------------


class TestCalendarListGBRouting(unittest.TestCase):
    def setUp(self):
        dc_utils._CALENDAR_MAP = {}

    def test_gb_all_in_bench_url_map(self):
        """CALENDAR_BENCH_URL_MAP must contain 'GB_ALL' mapped to '^FTSE'."""
        self.assertIn("GB_ALL", dc_utils.CALENDAR_BENCH_URL_MAP)
        self.assertEqual(dc_utils.CALENDAR_BENCH_URL_MAP["GB_ALL"], "^FTSE")

    def test_gb_startswith_guard(self):
        """'GB_ALL'.startswith('GB_') must be True so the Ticker branch is taken."""
        self.assertTrue("GB_ALL".startswith("GB_"))

    @patch("data_collector.utils.Ticker")
    def test_get_calendar_list_calls_ticker_with_ftse(self, mock_ticker_cls):
        """get_calendar_list('GB_ALL') must call Ticker('^FTSE').history(...)."""
        dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
        idx = pd.MultiIndex.from_tuples(
            [("^FTSE", d) for d in dates], names=["symbol", "date"]
        )
        mock_df = pd.DataFrame({"close": [7700.0, 7750.0]}, index=idx)
        mock_instance = MagicMock()
        mock_instance.history.return_value = mock_df
        mock_ticker_cls.return_value = mock_instance

        calendar = dc_utils.get_calendar_list("GB_ALL")

        mock_ticker_cls.assert_called_with("^FTSE")
        mock_instance.history.assert_called_with(interval="1d", period="max")
        self.assertEqual(len(calendar), 2)
        self.assertIsInstance(calendar[0], pd.Timestamp)


# ---------------------------------------------------------------------------
# Tests: YahooCollectorGB classes
# ---------------------------------------------------------------------------


class TestYahooCollectorGBClasses(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dc_utils._GB_SYMBOLS = None
        cls._sym_patch = patch(
            "data_collector.utils.requests.get",
            return_value=_make_screener_page(["AZN.L", "BP.L", "HSBA.L"]),
        )
        cls._sym_patch.start()
        import collector as col_mod

        cls.col = col_mod

    @classmethod
    def tearDownClass(cls):
        cls._sym_patch.stop()

    def setUp(self):
        dc_utils._GB_SYMBOLS = None

    def test_timezone_is_europe_london(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            obj = self.col.YahooCollectorGB1d(save_dir=tmpdir, start="2024-01-01", end="2024-01-10")
        self.assertEqual(obj._timezone, "Europe/London")

    def test_normalize_symbol_returns_uppercase(self):
        import tempfile
        from qlib.utils import code_to_fname

        with tempfile.TemporaryDirectory() as tmpdir:
            obj = self.col.YahooCollectorGB1d(save_dir=tmpdir, start="2024-01-01", end="2024-01-10")
        self.assertEqual(obj.normalize_symbol("AZN.L"), code_to_fname("AZN.L").upper())

    def test_gb1min_class_exists(self):
        self.assertTrue(hasattr(self.col, "YahooCollectorGB1min"))

    def test_normalize_gb1d_class_exists(self):
        self.assertTrue(hasattr(self.col, "YahooNormalizeGB1d"))

    def test_normalize_gb1d_extend_class_exists(self):
        self.assertTrue(hasattr(self.col, "YahooNormalizeGB1dExtend"))

    def test_normalize_gb1min_class_exists(self):
        self.assertTrue(hasattr(self.col, "YahooNormalizeGB1min"))


# ---------------------------------------------------------------------------
# Tests: Run class-name resolution for GB
# ---------------------------------------------------------------------------


class TestRunClassResolutionGB(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import collector as col_mod

        cls.col = col_mod

    def _make_run(self, region, interval):
        run = self.col.Run.__new__(self.col.Run)
        run.region = region
        run.interval = interval
        return run

    def test_collector_class_name_1d(self):
        self.assertEqual(self._make_run("GB", "1d").collector_class_name, "YahooCollectorGB1d")

    def test_collector_class_name_1min(self):
        self.assertEqual(self._make_run("GB", "1min").collector_class_name, "YahooCollectorGB1min")

    def test_normalize_class_name_1d(self):
        self.assertEqual(self._make_run("GB", "1d").normalize_class_name, "YahooNormalizeGB1d")

    def test_normalize_class_name_1min(self):
        self.assertEqual(self._make_run("GB", "1min").normalize_class_name, "YahooNormalizeGB1min")

    def test_all_gb_classes_resolvable_from_module(self):
        for name in [
            "YahooCollectorGB1d",
            "YahooCollectorGB1min",
            "YahooNormalizeGB1d",
            "YahooNormalizeGB1dExtend",
            "YahooNormalizeGB1min",
        ]:
            self.assertTrue(hasattr(self.col, name), f"Missing class: {name}")


if __name__ == "__main__":
    unittest.main()
