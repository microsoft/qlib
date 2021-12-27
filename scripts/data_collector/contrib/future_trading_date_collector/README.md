# Get future trading days

> `D.calendar(future=True)` will be used

## Requirements

```bash
pip install -r requirements.txt
```

## Collector Data

```bash
# parse instruments, using in qlib/instruments.
python future_trading_date_collector.py --qlib_dir ~/.qlib/qlib_data/cn_1d --freq day
```

## Parameters

- qlib_dir: qlib data directory
- freq: value from [`day`, `1min`], default `day`



