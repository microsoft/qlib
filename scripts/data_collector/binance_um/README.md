# Binance UM Perpetual Futures Collector (1min / 60min) for Qlib

This collector follows Qlib's official `scripts/data_collector` pattern:

- **History (bulk)**: download Binance public monthly ZIP klines from `data.binance.vision`
- **Live / incremental**: fetch klines from Binance USDⓈ-M Futures REST API (`/fapi/v1/klines`)
- **Normalize**: deduplicate, sort, and (optionally) build a 24/7 calendar for crypto
- **Dump**: convert normalized CSVs into Qlib `.bin` dataset via `scripts/dump_bin.py` (using `DumpDataAll`)

## Frequencies

- `1min`: Binance interval `1m`, Qlib dump freq `1min`
- `60min` (a.k.a. `1h`): Binance interval `1h`, Qlib dump freq **`60min`**
- `1d`: Binance interval `1d`, Qlib dump freq `day`

## Instrument naming

By default instruments are prefixed to avoid collisions:

- `binance_um.BTCUSDT`

The CSV file name is the same as the instrument (after `qlib.utils.code_to_fname`):

- `binance_um.BTCUSDT.csv`

## Data schema (Binance -> Qlib)

Binance Futures kline columns (REST & monthly ZIP CSV) use the same order:

`open_time, open, high, low, close, volume, close_time, quote_volume, trades, taker_buy_volume, taker_buy_quote_volume, ignore`

Normalized CSV columns (per instrument):

- `date`: UTC timestamp string, format `YYYY-MM-DD HH:MM:SS` (kline open time)
- `open, high, low, close`: mapped from kline cols 1-4
- `volume`: mapped from kline col 5 (base asset volume)
- `amount`: mapped from **kline col 7 (quote_volume)** (notional/turnover, UM perpetuals usually in USDT)
- `vwap`: computed as `amount / volume` when `volume > 0`, else NaN
- `trades`: mapped from col 8
- `taker_buy_volume`: mapped from col 9
- `taker_buy_amount`: mapped from col 10
- `symbol`: Qlib instrument name (e.g. `binance_um.BTCUSDT`)

## Usage

All commands are invoked via `fire`:

```bash
python qlib/scripts/data_collector/binance_um/collector.py --help
```

### A) Live / incremental (REST) collection

#### 1min

```bash
python qlib/scripts/data_collector/binance_um/collector.py download_data \
  --source_dir ~/.qlib/binance_um/source_1min \
  --normalize_dir ~/.qlib/binance_um/normalize_1min \
  --interval 1min \
  --start 2024-01-01 \
  --end 2024-02-01 \
  --symbols BTCUSDT,ETHUSDT \
  --delay 0.2
```

#### 60min (1h)

```bash
python qlib/scripts/data_collector/binance_um/collector.py download_data \
  --source_dir ~/.qlib/binance_um/source_60min \
  --normalize_dir ~/.qlib/binance_um/normalize_60min \
  --interval 60min \
  --start 2024-01-01 \
  --end 2024-06-01 \
  --symbols BTCUSDT,ETHUSDT \
  --delay 0.2
```

**Resume behavior**: if a per-symbol CSV already exists in `source_dir`, the collector will read its last `date` and continue from the next bar.

### B) History (monthly ZIP) download + convert

#### Download monthly ZIPs (1m or 1h)

```bash
python qlib/scripts/data_collector/binance_um/collector.py download_monthly_zip \
  --months 2023-11,2023-12,2024-01 \
  --raw_zip_dir ~/.qlib/binance_um/raw_zip \
  --zip_interval 1m \
  --symbols BTCUSDT,ETHUSDT
```

For hourly ZIPs:

```bash
python qlib/scripts/data_collector/binance_um/collector.py download_monthly_zip \
  --months 2023-11,2023-12,2024-01 \
  --raw_zip_dir ~/.qlib/binance_um/raw_zip \
  --zip_interval 1h \
  --symbols BTCUSDT,ETHUSDT
```

The downloader writes `manifest.json` to record `ok/missing/error` (404 is recorded as `missing` because many contracts listed later).

#### Convert ZIPs to per-symbol source CSV

```bash
python qlib/scripts/data_collector/binance_um/collector.py convert_monthly_zip_to_source \
  --raw_zip_dir ~/.qlib/binance_um/raw_zip \
  --source_dir ~/.qlib/binance_um/source_1min \
  --zip_interval 1m \
  --inst_prefix binance_um.
```

### C) Normalize

#### 1min normalize (24/7 calendar optional)

```bash
python qlib/scripts/data_collector/binance_um/collector.py normalize_data \
  --source_dir ~/.qlib/binance_um/source_1min \
  --normalize_dir ~/.qlib/binance_um/normalize_1min \
  --interval 1min \
  --fill_missing True
```

#### 60min normalize + optional fallback fill from 1min

If your 60min data is incomplete, you can fill missing hours from a 1min directory:

```bash
python qlib/scripts/data_collector/binance_um/collector.py normalize_data \
  --source_dir ~/.qlib/binance_um/source_60min \
  --normalize_dir ~/.qlib/binance_um/normalize_60min \
  --interval 60min \
  --fill_missing True \
  --fallback_1min_dir ~/.qlib/binance_um/source_1min
```

### D) Dump to Qlib `.bin`

#### 1min dataset

```bash
python qlib/scripts/data_collector/binance_um/collector.py dump_to_bin \
  --source_dir ~/.qlib/binance_um/source_1min \
  --normalize_dir ~/.qlib/binance_um/normalize_1min \
  --interval 1min \
  --qlib_dir ~/.qlib/qlib_data/binance_um_1min
```

#### 60min dataset

```bash
python qlib/scripts/data_collector/binance_um/collector.py dump_to_bin \
  --source_dir ~/.qlib/binance_um/source_60min \
  --normalize_dir ~/.qlib/binance_um/normalize_60min \
  --interval 60min \
  --qlib_dir ~/.qlib/qlib_data/binance_um_60min
```

## Optional: build 60min source CSV from 1min source CSV

This generates a 60min **source** directory by aggregating 1min:

```bash
python qlib/scripts/data_collector/binance_um/collector.py build_60min_from_1min \
  --source_1min_dir ~/.qlib/binance_um/source_1min \
  --target_60min_dir ~/.qlib/binance_um/source_60min_from_1min \
  --overwrite False
```

## Daily (1d) example (ZIP -> CSV -> normalize -> dump)

```bash
python qlib/scripts/data_collector/binance_um/collector.py download_monthly_zip \
  --months 2024-01 \
  --raw_zip_dir ~/.qlib/binance_um/raw_zip_1d \
  --zip_interval 1d \
  --symbols BTCUSDT

python qlib/scripts/data_collector/binance_um/collector.py convert_monthly_zip_to_source \
  --raw_zip_dir ~/.qlib/binance_um/raw_zip_1d \
  --source_dir ~/.qlib/binance_um/source_1d \
  --zip_interval 1d \
  --inst_prefix binance_um.

python qlib/scripts/data_collector/binance_um/collector.py normalize_data \
  --source_dir ~/.qlib/binance_um/source_1d \
  --normalize_dir ~/.qlib/binance_um/normalize_1d \
  --interval 1d \
  --fill_missing True

python qlib/scripts/data_collector/binance_um/collector.py dump_to_bin \
  --source_dir ~/.qlib/binance_um/source_1d \
  --normalize_dir ~/.qlib/binance_um/normalize_1d \
  --interval 1d \
  --qlib_dir ~/.qlib/qlib_data/binance_um_1d
```


