# Collect Point-in-Time Data

> *Please pay **ATTENTION** that the data is collected from [baostock](http://baostock.com) and the data might not be perfect. We recommend users to prepare their own data if they have high-quality dataset. For more information, users can refer to the [related document](https://qlib.readthedocs.io/en/latest/component/data.html#converting-csv-format-into-qlib-format)*

## Requirements

```bash
pip install -r requirements.txt
```

## Collector Data


### Download Quarterly CN Data

```bash
cd qlib/scripts/data_collector/pit/
# download from baostock.com
python collector.py download_data --source_dir ~/.qlib/stock_data/source/pit --start 2000-01-01 --end 2020-01-01 --interval quarterly
```

Downloading all data from the stock is very time-consuming. If you just want to run a quick test on a few stocks,  you can run the command below
```bash
python collector.py download_data --source_dir ~/.qlib/stock_data/source/pit --start 2000-01-01 --end 2020-01-01 --interval quarterly --symbol_regex "^(600519|000725).*"
```


### Normalize Data
```bash
python collector.py normalize_data --interval quarterly --source_dir ~/.qlib/stock_data/source/pit --normalize_dir ~/.qlib/stock_data/source/pit_normalized
```



### Dump Data into PIT Format

```bash
cd qlib/scripts
python dump_pit.py dump --data_path ~/.qlib/stock_data/source/pit_normalized --qlib_dir ~/.qlib/qlib_data/cn_data --interval quarterly
```
