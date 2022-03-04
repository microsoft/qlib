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
python collector.py download_data --source_dir ./csv_pit --start 2000-01-01 --end 2020-01-01 --interval quarterly
```

Downloading all data from the stock is very time consuming. If you just want run a quick test on a few stocks,  you can run the command below
``` bash
python collector.py download_data --source_dir ./csv_pit --start 2000-01-01 --end 2020-01-01 --interval quarterly --symbol_flt_regx "^(600519|000725).*"
```



### Dump Data into PIT Format

```bash
cd qlib/scripts
# data_collector/pit/csv_pit is the data you download just now.
python dump_pit.py dump --csv_path data_collector/pit/csv_pit --qlib_dir ~/.qlib/qlib_data/cn_data --interval quarterly
```
