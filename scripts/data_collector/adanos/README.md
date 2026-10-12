# Collect Daily US Retail Sentiment Factors From Adanos

This collector adds daily **retail sentiment** factors for US equities, based on the Adanos Market Sentiment API.

It is intended as an **alternative data** path for Qlib users who already maintain US price data and want to merge in daily sentiment features for factor research.

## What it downloads

For each symbol, the collector stores one CSV file with daily rows and these fields when available:

- `reddit_buzz`, `reddit_sentiment`, `reddit_mentions`
- `x_buzz`, `x_sentiment`, `x_mentions`, `x_avg_rank`
- `news_buzz`, `news_sentiment`, `news_mentions`
- `polymarket_buzz`, `polymarket_sentiment`, `polymarket_trade_count`
- `retail_buzz_avg`, `retail_sentiment_avg`, `retail_coverage`, `retail_alignment_score`

## Important limitation

The public Adanos stock detail endpoints expose a bounded historical lookback. This collector is therefore best used as:

1. an initial backfill for the most recent available window, and
2. a scheduled daily update to build a longer internal history over time.

## Requirements

```bash
pip install -r requirements.txt
```

## Typical workflow

1. Prepare or download US daily price CSVs with the existing Yahoo collector.
2. Download daily sentiment CSVs with this collector.
3. Normalize the sentiment CSVs.
4. Merge the sentiment fields into the normalized price CSVs.
5. Dump the merged CSVs into qlib `.bin` format.
6. Use `Alpha158AdanosUS` in a benchmark config.

## Download sentiment CSVs

```bash
python scripts/data_collector/adanos/collector.py download_data \
  --api_key <ADANOS_API_KEY> \
  --symbols AAPL,NVDA,TSLA,AMD \
  --source_dir ~/.qlib/stock_data/source/us_sentiment \
  --start 2025-10-01 \
  --end 2025-12-31
```

Or reuse an instruments file:

```bash
python scripts/data_collector/adanos/collector.py download_data \
  --api_key <ADANOS_API_KEY> \
  --instruments_path ~/.qlib/qlib_data/us_data/instruments/sp500.txt \
  --source_dir ~/.qlib/stock_data/source/us_sentiment \
  --start 2025-10-01 \
  --end 2025-12-31
```

## Normalize sentiment CSVs

```bash
python scripts/data_collector/adanos/collector.py normalize_data \
  --source_dir ~/.qlib/stock_data/source/us_sentiment \
  --normalize_dir ~/.qlib/stock_data/source/us_sentiment_nor
```

## Merge into normalized US price CSVs

```bash
python scripts/data_collector/adanos/collector.py merge_with_price_data \
  --price_dir ~/.qlib/stock_data/source/us_1d_nor \
  --sentiment_dir ~/.qlib/stock_data/source/us_sentiment_nor \
  --target_dir ~/.qlib/stock_data/source/us_1d_adanos
```

## Dump merged data into qlib format

```bash
python scripts/dump_bin.py dump_all \
  --data_path ~/.qlib/stock_data/source/us_1d_adanos \
  --qlib_dir ~/.qlib/qlib_data/us_data_adanos \
  --freq day \
  --exclude_fields date,symbol \
  --file_suffix .csv
```

## Use with the benchmark config

```bash
cd examples/benchmarks/LightGBM
qrun workflow_config_lightgbm_Alpha158Adanos_US.yaml
```
