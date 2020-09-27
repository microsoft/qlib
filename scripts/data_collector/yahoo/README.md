# Collect Data From Yahoo Finance

> *Please pay **ATTENTION** that the data is collected from [Yahoo Finance](https://finance.yahoo.com/lookup) and the data might not be perfect. We recommend users to prepare their own data if they have high-quality dataset. For more information, users can refer to the [related document](https://qlib.readthedocs.io/en/latest/component/data.html#converting-csv-format-into-qlib-format)*


>  **Examples of abnormal data**

- [SH000661](https://finance.yahoo.com/quote/000661.SZ/history?period1=1558310400&period2=1590796800&interval=1d&filter=history&frequency=1d)
- [SZ300144](https://finance.yahoo.com/quote/300144.SZ/history?period1=1557446400&period2=1589932800&interval=1d&filter=history&frequency=1d)

We have considered **STOCK PRICE ADJUSTMENT**, but some price series seem still very abnormal.

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
