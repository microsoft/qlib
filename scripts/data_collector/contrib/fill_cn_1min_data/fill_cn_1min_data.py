# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import fire
import qlib
import pandas as pd
from tqdm import tqdm
from qlib.data import D
from loguru import logger

CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent.parent.parent))
from data_collector.utils import generate_minutes_calendar_from_daily


def get_date_range(data_1min_dir: Path, max_workers: int = 16, date_field_name: str = "date"):
    csv_files = list(data_1min_dir.glob("*.csv"))
    min_date = None
    max_date = None
    with tqdm(total=len(csv_files)) as p_bar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for _file, _result in zip(csv_files, executor.map(pd.read_csv, csv_files)):
                if not _result.empty:
                    _dates = pd.to_datetime(_result[date_field_name])

                    _tmp_min = _dates.min()
                    min_date = min(min_date, _tmp_min) if min_date is not None else _tmp_min
                    _tmp_max = _dates.max()
                    max_date = max(max_date, _tmp_max) if max_date is not None else _tmp_max
                p_bar.update()
    return min_date, max_date


def get_symbols(data_1min_dir: Path):
    return list(map(lambda x: x.name[:-4].upper(), data_1min_dir.glob("*.csv")))


def fill_1min_using_1d(
    data_1min_dir: [str, Path],
    qlib_data_1d_dir: [str, Path],
    max_workers: int = 16,
    date_field_name: str = "date",
    symbol_field_name: str = "symbol",
):
    """Use 1d data to fill in the missing symbols relative to 1min

    Parameters
    ----------
    data_1min_dir: str
        1min data dir
    qlib_data_1d_dir: str
        1d qlib data(bin data) dir, from: https://qlib.readthedocs.io/en/latest/component/data.html#converting-csv-format-into-qlib-format
    max_workers: int
        ThreadPoolExecutor(max_workers), by default 16
    date_field_name: str
        date field name, by default date
    symbol_field_name: str
        symbol field name, by default symbol

    """
    data_1min_dir = Path(data_1min_dir).expanduser().resolve()
    qlib_data_1d_dir = Path(qlib_data_1d_dir).expanduser().resolve()

    min_date, max_date = get_date_range(data_1min_dir, max_workers, date_field_name)
    symbols_1min = get_symbols(data_1min_dir)

    qlib.init(provider_uri=str(qlib_data_1d_dir))
    data_1d = D.features(D.instruments("all"), ["$close"], min_date, max_date, freq="day")

    miss_symbols = set(data_1d.index.get_level_values(level="instrument").unique()) - set(symbols_1min)
    if not miss_symbols:
        logger.warning("More symbols in 1min than 1d, no padding required")
        return

    logger.info(f"miss_symbols  {len(miss_symbols)}: {miss_symbols}")
    tmp_df = pd.read_csv(list(data_1min_dir.glob("*.csv"))[0])
    columns = tmp_df.columns
    _si = tmp_df[symbol_field_name].first_valid_index()
    is_lower = tmp_df.loc[_si][symbol_field_name].islower()
    for symbol in tqdm(miss_symbols):
        if is_lower:
            symbol = symbol.lower()
        index_1d = data_1d.loc(axis=0)[symbol.upper()].index
        index_1min = generate_minutes_calendar_from_daily(index_1d)
        index_1min.name = date_field_name
        _df = pd.DataFrame(columns=columns, index=index_1min)
        if date_field_name in _df.columns:
            del _df[date_field_name]
        _df.reset_index(inplace=True)
        _df[symbol_field_name] = symbol
        _df["paused_num"] = 0
        _df.to_csv(data_1min_dir.joinpath(f"{symbol}.csv"), index=False)


if __name__ == "__main__":
    fire.Fire(fill_1min_using_1d)
