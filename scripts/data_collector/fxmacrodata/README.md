# Collect Data From FXMacroData

FXMacroData provides daily FX spot rates plus macroeconomic announcement,
release-calendar, and forecast data for currency-driven research. This collector
downloads those series into qlib-compatible CSV files and then uses qlib's
existing `dump_bin.py` script to convert them into qlib binary data.

The collector targets the canonical FXMacroData API base
`https://api.fxmacrodata.com/v1`.

## Requirements

```bash
pip install -r scripts/data_collector/fxmacrodata/requirements.txt
```

Set an API key when you need authenticated access:

```bash
export FXMACRODATA_API_KEY="<your key>"
```

`FXMD_API_KEY` is also supported. You can also pass `--api_key` to the collector.
Calendar, data-catalogue, and some USD evaluation endpoints are public. Broader
history, higher request budgets, non-USD announcement history, COT, commodities,
and other protected datasets require a subscribed API key. See
https://fxmacrodata.com/subscribe for plans and
https://fxmacrodata.com/documentation for endpoint coverage.

## Discover Available Macro Features

Use the catalogue command before building a feature set. It writes one row per
currency/indicator with source and coverage metadata where the API exposes it:

```bash
python scripts/data_collector/fxmacrodata/collector.py download_catalogue \
  --source_dir ~/.qlib/fxmacrodata/catalogue \
  --currencies usd,eur,gbp \
  --include_coverage true
```

The output file is `data_catalogue.csv`. Use its indicator slugs with
`download_macro_data --indicators ...`.

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

## Download Macro Announcement Features

Realized announcement values are available through the announcements dataset:

```bash
python scripts/data_collector/fxmacrodata/collector.py download_macro_data \
  --source_dir ~/.qlib/fxmacrodata/macro_source \
  --dataset announcements \
  --currencies usd \
  --indicators inflation,policy_rate,non_farm_payrolls \
  --start 2025-01-01 \
  --end 2026-01-01
```

Upcoming official release-calendar rows and forecast groups use the same command
with a different `--dataset`:

```bash
python scripts/data_collector/fxmacrodata/collector.py download_macro_data \
  --source_dir ~/.qlib/fxmacrodata/calendar_source \
  --dataset calendar \
  --currencies usd \
  --indicators inflation,policy_rate

python scripts/data_collector/fxmacrodata/collector.py download_macro_data \
  --source_dir ~/.qlib/fxmacrodata/prediction_source \
  --dataset predictions \
  --currencies usd \
  --indicators inflation
```

Macro files use symbols such as `usd_inflation` and include numeric features
such as `value`, `actual`, `consensus`, `forecast`, `surprise`, `prediction`,
`prediction_count`, `announcement_datetime`, and `is_future`.

For broader multi-currency history, pass an authenticated API key and select the
currencies/indicators from `download_catalogue`:

```bash
python scripts/data_collector/fxmacrodata/collector.py download_macro_data \
  --source_dir ~/.qlib/fxmacrodata/macro_source \
  --dataset announcements \
  --currencies usd,eur,gbp,jpy \
  --indicators inflation,policy_rate,gdp \
  --start 2015-01-01 \
  --end 2026-01-01 \
  --api_key "$FXMACRODATA_API_KEY"
```

Normalize the macro CSVs before dumping them:

```bash
python scripts/data_collector/fxmacrodata/collector.py normalize_macro_data \
  --source_dir ~/.qlib/fxmacrodata/macro_source \
  --normalize_dir ~/.qlib/fxmacrodata/macro_normalize
```

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
macro = D.features(["usd_inflation"], ["$value", "$forecast", "$announcement_datetime"], freq="day")
```

## Point-In-Time Notes

Use the `announcements` dataset for historical training targets and realized
macro features. `calendar` and `predictions` are useful for event-risk research,
but they may include future scheduled rows. For backtests, filter on
`$is_future == 0` or join features using `announcement_datetime` so the model
only sees information that would have been known at the simulated time.

Subscribed users can combine FX spot data, realized announcements, release
calendars, forecasts, COT, commodities, and additional macro endpoints through
the FXMacroData API. Start at https://fxmacrodata.com/subscribe when the public
evaluation endpoints do not cover the required currencies, history, or volume.
