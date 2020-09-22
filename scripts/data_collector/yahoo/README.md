# Collect Data From Yahoo Finance

## Requirements

```bash
pip install -r requirements.txt
```

## Collector Data

### Download data -> Normalize data -> Dump data
```bash
python collector.py collector_data --source_dir ~/.qlib/stock_data/source --normalize_dir ~/.qlib/stock_data/normalize_dir --qlib_dir ~/.qlib/stock_data/qlib_data
```

### Download Data From Yahoo Finance

```bash
python collector.py download_data --source_dir ~/.qlib/stock_data/source
```

### Normalize Yahoo Finance Data

```bash
python collector.py normalize_data --source_dir ~/.qlib/stock_data/source --normalize_dir ~/.qlib/stock_data/normalize
```

### Manual Ajust Yahoo Finance Data

```bash
python collector.py manual_adj_data --normalize_dir ~/.qlib/stock_data/normalize
```

### Dump Yahoo Finance Data

```bash
python collector.py dump_data --normalize_dir ~/.qlib/stock_data/normalize_dir --qlib_dir ~/.qlib/stock_data/qlib_data
```