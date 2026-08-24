# Collect X market-attention data with Xquik

This collector turns X (formerly Twitter) search results into daily numeric
features for Qlib. It uses the published Xquik REST API and supports Qlib's
Python 3.8 baseline.

The collector stores counts only. It does not store post text or author
identifiers.

## Requirements

Install the collector dependencies:

```bash
pip install -r scripts/data_collector/xquik/requirements.txt
```

Set an [Xquik API key](https://docs.xquik.com):

```bash
export X_TWITTER_SCRAPER_API_KEY="your-api-key"
```

The collector reads the key only from the environment. It never writes the key
to collected data.

## Collect daily features

Run the command from the Qlib repository root:

```bash
python scripts/data_collector/xquik/collector.py download_data \
  --source_dir ~/.qlib/xquik_data/source \
  --symbols AAPL,MSFT \
  --start 2026-08-01 \
  --end 2026-08-08 \
  --interval 1d
```

`start` is inclusive. `end` is exclusive. Both values must be UTC day
boundaries. The default query template is `${symbol}`, which searches each
symbol as an X cashtag.

Change the query while keeping the `{symbol}` placeholder:

```bash
python scripts/data_collector/xquik/collector.py download_data \
  --source_dir ~/.qlib/xquik_data/source \
  --symbols AAPL,MSFT \
  --start 2026-08-01 \
  --end 2026-08-08 \
  --query_template '{symbol} earnings' \
  --language en
```

Replies and reposts are excluded by default. Use `--include_replies True` or
`--include_retweets True` when they belong in the research question.

The collector requests coverage for one UTC day at a time. It rejects a day
when Xquik reports incomplete, stalled, failed, limited, or truncated coverage.
This prevents a partial response from becoming a silent zero or weak signal.

## Feature definitions

Each symbol and UTC day produces these numeric fields:

| Feature | Definition |
| --- | --- |
| `post_count` | Unique posts returned for the query |
| `author_count` | Unique available author IDs |
| `observed_like_count` | Likes reported at collection time |
| `observed_retweet_count` | Reposts reported at collection time |
| `observed_reply_count` | Replies reported at collection time |
| `observed_quote_count` | Quotes reported at collection time |
| `observed_bookmark_count` | Bookmarks reported at collection time |
| `observed_view_count` | Views reported at collection time |
| `observation_time` | Collection time as Unix seconds |

A day without matching posts produces a row of zeros. X may report zero when a
metric is unavailable.

## Avoid lookahead bias

Engagement totals describe collection time, not post creation time. A
historical backfill can include engagement that happened after a model's cutoff.
Deleted or unavailable posts may also be absent from a later backfill.

For point-in-time research, collect each completed day before the next model
cutoff. Retain `observation_time` and archive the raw CSV files. Do not treat a
later backfill as a historical snapshot.

## Normalize and load the data

Normalize the collected CSV files:

```bash
python scripts/data_collector/xquik/collector.py normalize_data \
  --source_dir ~/.qlib/xquik_data/source \
  --normalize_dir ~/.qlib/xquik_data/normalized \
  --interval 1d
```

Convert them into Qlib data:

```bash
python scripts/dump_bin.py dump_all \
  --data_path ~/.qlib/xquik_data/normalized \
  --qlib_dir ~/.qlib/qlib_data/xquik \
  --freq day \
  --exclude_fields date,symbol
```

Read the features through Qlib:

```python
import qlib
from qlib.data import D

qlib.init(provider_uri="~/.qlib/qlib_data/xquik")
features = D.features(
    D.instruments("all"),
    ["$post_count", "$author_count", "$observed_view_count"],
    freq="day",
)
```
