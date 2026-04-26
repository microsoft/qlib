import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.joinpath("scripts")))

from convert_rqalpha_bundle import (  # noqa: E402
    RQAlphaBundleConverter,
    rqalpha_code_to_qlib_code,
    rqalpha_datetime_to_timestamp,
)


def test_rqalpha_code_to_qlib_code():
    assert rqalpha_code_to_qlib_code("000001.XSHE") == "SZ000001"
    assert rqalpha_code_to_qlib_code("600000.XSHG") == "SH600000"


def test_rqalpha_datetime_to_timestamp():
    assert rqalpha_datetime_to_timestamp(20050104000000) == pd.Timestamp("2005-01-04")


def test_normalize_dataset():
    rows = [
        (20050105000000, 2.0, 2.1, 2.2, 1.9, 100.0, 220.0),
        (20050104000000, 1.0, 1.1, 1.2, 0.9, 90.0, 180.0),
    ]
    fields = ["open", "close", "high", "low", "volume", "vwap", "factor"]
    dataset = pd.DataFrame.from_records(
        rows, columns=["datetime", "open", "close", "high", "low", "volume", "total_turnover"]
    ).to_records(index=False)
    factor_rows = pd.DataFrame.from_records(
        [(0, 1.0), (20050105000000, 2.0)],
        columns=["start_date", "ex_cum_factor"],
    ).to_records(index=False)

    df = RQAlphaBundleConverter._normalize_dataset(dataset, fields, "000001.XSHE", factor_rows)

    assert list(df.columns) == ["date", "symbol", *fields]
    assert df["symbol"].unique().tolist() == ["SZ000001"]
    assert df["date"].tolist() == [pd.Timestamp("2005-01-04"), pd.Timestamp("2005-01-05")]
    assert df["close"].tolist() == [1.1, 4.2]
    assert df["vwap"].tolist() == [2.0, 4.4]
    assert df["volume"].tolist() == [90.0, 50.0]
    assert df["factor"].tolist() == [1.0, 2.0]
