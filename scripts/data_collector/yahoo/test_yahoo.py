# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import unittest
from collector import YahooCollector,YahooCollectorCN
import sys
from pathlib import Path

CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent.parent))


class Test_Yahoo(unittest.TestCase):
    def to_str(self, obj):
        return "".join(str(obj).split())

    def check_same(self, a, b):
        self.assertEqual(self.to_str(a), self.to_str(b))

    def to_symbol(self, str):
        return ".".join('%s' % id for id in (str[2:], str[:2]))

    def test_nomal(self):
        self.instruments = "sz000017"
        self.interval = "1d"
        self.start = "2021-06-11"
        self.end = "2021-06-20"
        self.fields = ["low", "open", "high", "close", "volume", "adjclose"]
        symbol = self.to_symbol(self.instruments)
        yahoo_data = YahooCollector.get_data_from_remote(symbol=symbol,
                                                         interval=self.interval,
                                                         start=self.start,
                                                         end=self.end)
        order = ["symbol", "date", "high", "volume", "low", "close", "open", "adjclose"]
        yahoo_data = yahoo_data[order]
        data = """
      symbol        date  high   volume   low  close  open  adjclose
0  000017.sz  2021-06-11  3.73  9320519  3.47   3.72  3.50      3.72
1  000017.sz  2021-06-15  3.67  5799953  3.53   3.53  3.64      3.53
2  000017.sz  2021-06-16  3.56  5747865  3.42   3.45  3.50      3.45
3  000017.sz  2021-06-17  3.52  4421531  3.40   3.50  3.45      3.50
4  000017.sz  2021-06-18  3.65  3842452  3.47   3.54  3.47      3.54
          """
        self.check_same(data, yahoo_data)

    def test_delisted(self):
        instruments = ["sz000018"]
        symbol = self.to_symbol(instruments)
        yahoo_data = YahooCollector.get_data_from_remote(symbol=symbol,
                                                         interval=self.interval,
                                                         start=self.start,
                                                         end=self.end)
        data = None
        self.check_same(data, yahoo_data)

    def test_normalize_symbol(self):
        normalize_symbol = YahooCollectorCN().normalize_symbol("sz000017")
        symbol = self.instruments
        self.check_same(normalize_symbol, symbol)

    def test_get_instrument_list(self):
        symbols_list=['000001.sz', '000002.sz', '000004.sz', '000005.sz', '000006.sz', '000007.sz', '000008.sz', '000009.sz', '000010.sz']
        instrument_list = YahooCollectorCN().get_instrument_list()
        instrument_list = instrument_list[:9]
        self.check_same(symbols_list, instrument_list)

    if __name__ == "__main__":
        unittest.main()
