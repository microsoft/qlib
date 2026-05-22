# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
from pathlib import Path

import pandas as pd


sys.path.append(str(Path(__file__).resolve().parent.parent.joinpath("scripts")))

from data_collector.base import BaseCollector


class DummyCollector(BaseCollector):
    def get_instrument_list(self):
        return []

    def normalize_symbol(self, symbol: str):
        return symbol

    def get_data(self, symbol: str, interval: str, start_datetime: pd.Timestamp, end_datetime: pd.Timestamp):
        return pd.DataFrame()


def test_base_collector_supports_default_5min_date_range(tmp_path):
    collector = DummyCollector(save_dir=tmp_path, interval="5min")

    assert collector.start_datetime == DummyCollector.DEFAULT_START_DATETIME_5MIN
    assert collector.end_datetime == DummyCollector.DEFAULT_END_DATETIME_5MIN
