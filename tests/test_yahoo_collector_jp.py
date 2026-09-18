# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
import importlib as stdlib_importlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT_DIR.joinpath("scripts")

sys.path.insert(0, str(SCRIPTS_DIR))

from data_collector import utils as dc_utils  # noqa: E402
from data_collector.yahoo import collector as yahoo_collector  # noqa: E402


class TestYahooCollectorJP(unittest.TestCase):
    def setUp(self):
        dc_utils._JP_SYMBOLS = None  # pylint: disable=W0212
        dc_utils._CALENDAR_MAP.pop("JP_ALL", None)  # pylint: disable=W0212

    @staticmethod
    def _import_module_side_effect(module_name: str):
        if module_name == "collector":
            return yahoo_collector
        return stdlib_importlib.import_module(module_name)

    def test_extract_jp_prime_symbols_with_etf_etn(self):
        source_df = pd.DataFrame(
            {
                "コード": ["7203", "6758", "1301", "9432", "1489", "1489"],
                "市場・商品区分": [
                    "プライム（内国株式）",
                    "スタンダード（内国株式）",
                    "プライム（内国株式）",
                    "プライム（外国株式）",
                    "ETF・ETN",
                    "ＥＴＦ・ＥＴＮ",
                ],
            }
        )

        symbols = dc_utils._extract_jp_prime_symbols(source_df)  # pylint: disable=W0212
        self.assertEqual(symbols, ["1301.T", "1489.T", "7203.T"])

    def test_extract_jp_prime_symbols_missing_columns(self):
        with self.assertRaisesRegex(ValueError, "stock code column"):
            dc_utils._extract_jp_prime_symbols(pd.DataFrame({"市場・商品区分": ["プライム（内国株式）"]}))  # pylint: disable=W0212

        with self.assertRaisesRegex(ValueError, "market classification column"):
            dc_utils._extract_jp_prime_symbols(pd.DataFrame({"コード": ["7203"]}))  # pylint: disable=W0212

    def test_get_jp_stock_symbols_from_jpx(self):
        source_df = pd.DataFrame(
            {
                "コード": ["7203", "1301", "0001", "8306", "1489"],
                "市場・商品区分": [
                    "プライム（内国株式）",
                    "プライム（内国株式）",
                    "ETF・ETN",
                    "プライム（内国株式）",
                    "ＥＴＦ・ＥＴＮ",
                ],
            }
        )

        class _Resp:
            status_code = 200
            content = b"dummy"

        with patch("data_collector.utils.requests.get", return_value=_Resp()) as mock_get:
            with patch("data_collector.utils.pd.read_excel", return_value=source_df) as mock_read_excel:
                symbols = dc_utils.get_jp_stock_symbols()
                symbols_cached = dc_utils.get_jp_stock_symbols()

        self.assertEqual(symbols, ["0001.T", "1301.T", "1489.T", "7203.T", "8306.T"])
        self.assertEqual(symbols_cached, ["0001.T", "1301.T", "1489.T", "7203.T", "8306.T"])
        self.assertEqual(mock_get.call_count, 1)
        self.assertEqual(mock_read_excel.call_count, 1)

    def test_run_class_resolution_for_jp_1d(self):
        with patch("data_collector.base.importlib.import_module", side_effect=self._import_module_side_effect):
            with tempfile.TemporaryDirectory() as tmp_dir:
                run = yahoo_collector.Run(
                    source_dir=tmp_dir,
                    normalize_dir=tmp_dir,
                    max_workers=1,
                    interval="1d",
                    region="JP",
                )

        self.assertEqual(run.collector_class_name, "YahooCollectorJP1d")
        self.assertEqual(run.normalize_class_name, "YahooNormalizeJP1d")
        self.assertIs(getattr(yahoo_collector, run.collector_class_name), yahoo_collector.YahooCollectorJP1d)
        self.assertIs(getattr(yahoo_collector, run.normalize_class_name), yahoo_collector.YahooNormalizeJP1d)

    def test_run_jp_1min_is_not_supported(self):
        with patch("data_collector.base.importlib.import_module", side_effect=self._import_module_side_effect):
            with tempfile.TemporaryDirectory() as tmp_dir:
                run = yahoo_collector.Run(
                    source_dir=tmp_dir,
                    normalize_dir=tmp_dir,
                    max_workers=1,
                    interval="1min",
                    region="JP",
                )
                with self.assertRaisesRegex(ValueError, "JP region does not support 1min data"):
                    run.download_data(start="2024-01-01", end="2024-01-05")
                with self.assertRaisesRegex(ValueError, "JP region does not support 1min data"):
                    run.normalize_data(qlib_data_1d_dir=tmp_dir)

    def test_get_calendar_list_jp_normalizes_timezone_values(self):
        multi_index = pd.MultiIndex.from_arrays(
            [
                ["^N225", "^N225", "^N225"],
                [
                    pd.Timestamp("2024-01-05"),
                    pd.Timestamp("2024-01-04 10:26:15+09:00"),
                    pd.Timestamp("2024-01-05 11:30:00+09:00"),
                ],
            ],
            names=["symbol", "date"],
        )
        history_df = pd.DataFrame({"close": [1, 2, 3]}, index=multi_index)

        class FakeTicker:
            def __init__(self, *args, **kwargs):
                pass

            def history(self, *args, **kwargs):
                return history_df

        with patch("data_collector.utils.Ticker", FakeTicker):
            calendar = dc_utils.get_calendar_list("JP_ALL")

        self.assertEqual(calendar, [pd.Timestamp("2024-01-04"), pd.Timestamp("2024-01-05")])

    def test_update_data_to_bin_jp_skips_index_components(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            qlib_dir = Path(tmp_dir).joinpath("qlib_data")
            qlib_dir.joinpath("calendars").mkdir(parents=True)
            qlib_dir.joinpath("calendars/day.txt").write_text("2024-01-04\n2024-01-05\n", encoding="utf-8")

            with patch("data_collector.base.importlib.import_module", side_effect=self._import_module_side_effect):
                run = yahoo_collector.Run(
                    source_dir=Path(tmp_dir).joinpath("source"),
                    normalize_dir=Path(tmp_dir).joinpath("normalize"),
                    max_workers=1,
                    interval="1d",
                    region="JP",
                )

            with patch("data_collector.yahoo.collector.exists_qlib_data", return_value=True):
                with patch.object(yahoo_collector.Run, "download_data", return_value=None) as mock_download:
                    with patch.object(yahoo_collector.Run, "normalize_data_1d_extend", return_value=None) as mock_normalize_ext:
                        with patch("data_collector.yahoo.collector.DumpDataUpdate") as mock_dump_cls:
                            with patch("data_collector.yahoo.collector.importlib.import_module") as mock_import:
                                run.update_data_to_bin(qlib_data_1d_dir=str(qlib_dir), end_date="2024-01-06")

        mock_download.assert_called_once_with(
            delay=1, start="2024-01-04", end="2024-01-06", check_data_length=None, limit_nums=None
        )
        mock_normalize_ext.assert_called_once()
        self.assertEqual(Path(mock_normalize_ext.call_args.args[0]).resolve(), qlib_dir.resolve())
        mock_dump_cls.return_value.dump.assert_called_once()
        mock_import.assert_not_called()


if __name__ == "__main__":
    unittest.main()
