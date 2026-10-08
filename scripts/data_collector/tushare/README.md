# TuShare Daily Data Collector

Collect CN daily equity data from TuShare, normalize to qlib CSV schema, and dump to qlib bin format. Supports full builds and incremental updates.

## Requirements
- Python venv is recommended; install dependencies:
  ```bash
  python -m pip install tushare plotly torch
  ```
- Set `TUSHARE_TOKEN` (e.g., put in `.env` then `export $(cat .env | xargs)`).
- Default qlib output: `~/.qlib/qlib_data/cn_data`.

## Quick Start (one-shot pipeline)
Download → normalize → dump in a single command:
```bash
python qlib/scripts/data_collector/tushare/collector.py pipeline \
  --source_dir ./tmp/tushare_raw \
  --normalize_dir ./tmp/tushare_norm \
  --qlib_dir ~/.qlib/qlib_data/cn_data \
  --start 2010-01-01 --end 2024-12-31 \
  --token "$TUSHARE_TOKEN"
```

## Step-by-Step
1) Download raw TuShare daily data to CSV:
```bash
python qlib/scripts/data_collector/tushare/collector.py download_data \
  --source_dir ./tmp/tushare_raw \
  --start 2020-01-01 --end 2020-12-31 \
  --token "$TUSHARE_TOKEN"
```
2) Normalize to qlib-ready CSVs (factor-adjusted prices, volume back-adjusted, symbols normalized):
```bash
python qlib/scripts/data_collector/tushare/collector.py normalize_data \
  --source_dir ./tmp/tushare_raw \
  --normalize_dir ./tmp/tushare_norm
```
3) Dump normalized CSVs to qlib bin format:
```bash
python qlib/scripts/data_collector/tushare/collector.py dump_to_bin \
  --normalize_dir ./tmp/tushare_norm \
  --qlib_dir ~/.qlib/qlib_data/cn_data \
  --mode all
```

## Incremental Update
Update an existing day-level qlib directory with fresh TuShare data:
```bash
python qlib/scripts/data_collector/tushare/collector.py update_data_to_bin \
  --qlib_data_1d_dir ~/.qlib/qlib_data/cn_data \
  --end_date 2024-12-31
```
- Starts from the last trading date in `calendars/day.txt` and only dumps newer rows.
- Reruns `download_data` + `normalize_data` internally and writes incremental bins.

## Validate a qlib Directory
```bash
python - <<'PY'
from qlib.scripts.data_collector.tushare.collector import validate_qlib_dir
print(validate_qlib_dir("~/.qlib/qlib_data/cn_data", freq="day"))
PY
```
Returns a dict; values are `None` when calendars, instruments, and feature bins are present.

## Notes
- Interval: currently 1d only.
- Required columns fetched: `ts_code`, `trade_date`, `open/high/low/close`, `vol`, `amount`, `adj_factor`.
- Prices are forward-adjusted by normalized `factor`; volume is back-adjusted by the same factor.
- Symbols are mapped from `000001.SZ` → `sz000001` to match qlib conventions.
- `save_instrument` deduplicates by date so reruns will not create duplicate rows.

