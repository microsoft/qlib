# US Fundamental Data Collector (Route 1.5)

Collect fundamental factors for US stocks using **Yahoo Finance** (free) + **SEC EDGAR** filing dates (free) to avoid look-ahead bias.

## Architecture

```
Yahoo Finance (yahooquery)         SEC EDGAR
  income_statement()                 CIK submissions API
  balance_sheet()                    → filing dates only
  cash_flow()                        (lightweight, no XBRL parsing)
        │                                    │
        ▼                                    ▼
  yahoo_fundamental.py              edgar_filing_dates.py
  (quarterly financials)            (when each 10-Q/10-K was filed)
        │                                    │
        └──────────────┬─────────────────────┘
                       ▼
               build_factors.py
               ├── merge on (symbol, reportDate)
               ├── use filingDate as availableDate (no look-ahead!)
               ├── compute factors (ROE, EP, Growth, ...)
               ├── forward-fill to daily frequency
               └── output per-symbol CSVs
                       │
                       ▼
              dump_bin.py dump_all
              (existing Qlib tool)
                       │
                       ▼
              Qlib binary features:
              features/AAPL/roe.day.bin
              features/AAPL/roa.day.bin
              features/AAPL/netincome.day.bin
              ...
```

## Quick Start

### Step 1: Prepare symbol list

```bash
# Use existing Qlib US data instrument list, or create your own
echo -e "AAPL\nMSFT\nGOOGL\nAMZN\nMETA\nNVDA\nTSLA" > symbols.txt
```

### Step 2: Collect Yahoo Finance fundamental data

```bash
python yahoo_fundamental.py collect_from_file \
    --symbol_file symbols.txt \
    --save_dir ./yahoo_data \
    --start 2018-01-01 \
    --delay 0.5
```

### Step 3: Collect SEC EDGAR filing dates

```bash
python edgar_filing_dates.py fetch_from_file \
    --symbol_file symbols.txt \
    --save_path ./edgar_filing_dates.csv \
    --delay 0.15
```

### Step 4: Build daily factor CSVs

```bash
python build_factors.py build \
    --yahoo_data_path ./yahoo_data/_all_fundamentals.csv \
    --edgar_data_path ./edgar_filing_dates.csv \
    --qlib_dir ~/.qlib/qlib_data/us_data \
    --output_dir ./fundamental_daily \
    --start 2018-01-01
```

### Step 5: Dump to Qlib binary format

```bash
# IMPORTANT: Use dump_update (not dump_all) to ADD fundamental features
# to an existing Qlib dataset that already has OHLCV data
python ../../../dump_bin.py dump_update \
    --data_path ./fundamental_daily \
    --qlib_dir ~/.qlib/qlib_data/us_data \
    --freq day \
    --exclude_fields symbol,date
```

### Step 6: Run a model with fundamental factors

```bash
cd ../../examples/us_fundamental
python -m qlib.workflow -c workflow_config.yaml
```

Or use the handler directly in Python:

```python
from qlib.contrib.data.handler_us import USAlphaFundamental

handler = USAlphaFundamental(
    instruments="sp500",
    start_time="2018-01-01",
    end_time="2024-12-31",
    fit_start_time="2018-01-01",
    fit_end_time="2022-12-31",
)
```

## Without SEC EDGAR (Simpler but Less Accurate)

If you want to skip the SEC EDGAR step, you can use a conservative fallback
lag. The `build_factors.py` script will add 90 days to each report period
date, which is safe but means you'll use data slightly later than necessary:

```bash
python build_factors.py build \
    --yahoo_data_path ./yahoo_data/_all_fundamentals.csv \
    --qlib_dir ~/.qlib/qlib_data/us_data \
    --output_dir ./fundamental_daily \
    --fallback_lag_days 90
```

## Available Factors

| Category | Factor | Description |
|----------|--------|-------------|
| Quality | `roe` | Return on Equity |
| Quality | `roa` | Return on Assets |
| Quality | `gross_margin` | Gross Profit / Revenue |
| Quality | `accruals` | (NI - OCF) / Assets (earnings quality) |
| Growth | `revenue_yoy` | Revenue growth YOY |
| Growth | `earnings_yoy` | Earnings growth YOY |
| Leverage | `debt_to_equity` | Total Debt / Equity |
| Value* | `netincome` | Used by handler as `$netincome/$close` |
| Value* | `totalrevenue` | Used by handler as `$totalrevenue/$close` |
| Value* | `freecashflow` | Used by handler as `$freecashflow/$close` |
| Value* | `stockholdersequity` | Used by handler as `$stockholdersequity/$close` |
| Value* | `ebitda` | Used by handler as `$ebitda/$close` |

*Value factors are computed as price-relative ratios in the handler, not in the CSV.

## Available Handlers

| Handler | Features | Use Case |
|---------|----------|----------|
| `USAlpha158` | ~170 tech factors | When you only have OHLCV data |
| `USFundamental` | ~12 fundamental factors | When you only want fundamentals |
| `USAlphaFundamental` | ~182 combined | **Recommended** for best results |

## Limitations

- **Yahoo Finance data depth**: Only ~4 years of quarterly data available
- **Filing date accuracy**: SEC EDGAR API returns recent filings; very old
  filings may not be available, in which case the fallback lag is used
- **No analyst estimates**: Yahoo Finance free tier doesn't provide consensus
  estimates or earnings surprises
