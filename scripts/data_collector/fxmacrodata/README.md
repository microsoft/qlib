# Collect Data From FXMacroData

FXMacroData provides daily FX spot-rate series for currency pairs such as `EUR/USD`.
This collector downloads those series into qlib-compatible CSV files and then uses
qlib's existing `dump_bin.py` script to convert them into qlib binary data.

## Requirements

```bash
pip install -r scripts/data_collector/fxmacrodata/requirements.txt
```

Set an API key when you need authenticated access:

```bash
export FXMACRODATA_API_KEY="<your key>"
```

`FXMD_API_KEY` is also supported. You can also pass `--api_key` to the collector.

## Download FX Data

```bash
python scripts/data_collector/fxmacrodata/collector.py download_data \
  --source_dir ~/.qlib/fxmacrodata/source \
  --start 2024-01-01 \
  --end 2024-03-01 \
  --pairs EURUSD,GBPUSD,USDJPY
```

Supported pair formats include `EURUSD`, `EUR/USD`, `EUR-USD`, `EUR_USD`, and
Yahoo-style `EURUSD=X`. The collector currently supports daily data only.

## Normalize Data

```bash
python scripts/data_collector/fxmacrodata/collector.py normalize_data \
  --source_dir ~/.qlib/fxmacrodata/source \
  --normalize_dir ~/.qlib/fxmacrodata/normalize
```

FX spot data is shaped with `open`, `high`, `low`, and `close` equal to the daily
spot rate. `volume` is set to `0`, `factor` is set to `1`, and `change` is the
daily percentage change in `close`.

## Dump To qlib Format

```bash
python scripts/dump_bin.py dump_all \
  --data_path ~/.qlib/fxmacrodata/normalize \
  --qlib_dir ~/.qlib/qlib_data/fxmacrodata \
  --freq day \
  --exclude_fields date,symbol \
  --file_suffix .csv
```

## Use The Data

```python
import qlib
from qlib.data import D

qlib.init(provider_uri="~/.qlib/qlib_data/fxmacrodata", region="us")
df = D.features(["eurusd", "gbpusd"], ["$close", "$change"], freq="day")
```
