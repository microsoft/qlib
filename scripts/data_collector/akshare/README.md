
- [Collector Data](#collector-data)
  - [Collector *AKShare* data to qlib](#collector-akshare-data-to-qlib)
- [Using qlib data](#using-qlib-data)


# Collect Data From AKShare (A-share)

> This collector fetches China A-share daily OHLCV data via [AKShare](https://github.com/akfamily/akshare). It covers all Shanghai and Shenzhen stocks (excluding B-shares).

## Requirements

```bash
pip install -r requirements.txt
```

## Collector Data

### Collector *AKShare* data to qlib

> Collect A-share daily data and dump into `qlib` format.

1. Download data to csv: `python scripts/data_collector/akshare/collector.py download_data`

   This downloads raw OHLCV data from AKShare to a local directory (one CSV per symbol).

   - parameters:
     - `source_dir`: save directory
     - `start`: start datetime, by default *"2000-01-01"*
     - `end`: end datetime, by default today
     - `delay`: `time.sleep(delay)`, by default *0.5*
     - `max_workers`: number of concurrent workers, by default *1*
     - `max_collector_count`: number of retries for failed symbols, by default *2*
     - `check_data_length`: minimum row count per symbol, by default `None`
     - `limit_nums`: limit number of symbols (for debugging), by default `None*
     - `symbols`: comma-separated stock codes, e.g. `"600519,000001"`. If omitted, auto-discovers all A-shares
     - `symbol_file`: path to a text file with one code per line
     - `adjust`: price adjustment, value from [`qfq`, `hfq`, `""`], by default `qfq`
   - examples:
     ```bash
     # all A-shares, daily, forward-adjusted
     python collector.py download_data --source_dir ~/.qlib/stock_data/source/akshare_data --start 2020-01-01 --end 2024-12-31 --delay 0.5

     # specific symbols only
     python collector.py download_data --source_dir ~/.qlib/stock_data/source/akshare_data --symbols "600519,000001,300750" --start 2024-01-01

     # from a symbol file, with no adjustment
     python collector.py download_data --source_dir ~/.qlib/stock_data/source/akshare_data --symbol_file symbols.txt --adjust ""
     ```

2. Normalize data: `python scripts/data_collector/akshare/collector.py normalize_data`

   This deduplicates, sorts by date, and aligns to the A-share trading calendar.

   - parameters:
     - `source_dir`: csv directory
     - `normalize_dir`: result directory
     - `max_workers`: number of concurrent workers, by default *1*
     - `date_field_name`: date column name, by default `date`
     - `symbol_field_name`: symbol column name, by default `symbol`
     - `end_date`: last date to include (inclusive), by default `None`
   - examples:
     ```bash
     python collector.py normalize_data --source_dir ~/.qlib/stock_data/source/akshare_data --normalize_dir ~/.qlib/stock_data/source/akshare_1d_nor
     ```

3. Dump data: `python scripts/dump_bin.py dump_all`

   Convert normalized CSV to qlib binary format.

   - parameters:
     - `data_path`: normalize result directory
     - `qlib_dir`: qlib data directory
     - `freq`: transaction frequency, by default `day`
     - `max_workers`: number of threads, by default *16*
     - `exclude_fields`: fields not dumped, by default `""`
   - examples:
     ```bash
     python scripts/dump_bin.py dump_all --data_path ~/.qlib/stock_data/source/akshare_1d_nor --qlib_dir ~/.qlib/qlib_data/akshare_data --freq day --exclude_fields date,symbol
     ```

## Using qlib data

  ```python
  import qlib
  from qlib.data import D

  qlib.init(provider_uri="~/.qlib/qlib_data/akshare_data", region="cn")
  df = D.features(D.instruments("all"), ["$close"], freq="day")
  ```
