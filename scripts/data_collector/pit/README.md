# Collect Point-in-Time Data

> *Please pay **ATTENTION** that the data is collected from [baostock](http://baostock.com) and the data might not be perfect. We recommend users to prepare their own data if they have high-quality dataset. For more information, users can refer to the [related document](https://qlib.readthedocs.io/en/latest/component/data.html#converting-csv-format-into-qlib-format)*

## Requirements

```bash
pip install -r requirements.txt
```

## Collector Data


### Download Quarterly CN Data

```bash

# download from baostock.com
python collector.py download_data --source_dir /data1/v-xiabi/qlib/pit/csv_2 --start 2000-01-01 --end 2020-01-01 --interval quarterly

```

### Dump Data into PIT Format

cd qlib/scripts
python dump_pit.py dump --csv_path /data1/v-xiabi/qlib/pit/csv_2 --qlib_dir ~/.qlib/qlib_data/cn_data --interval quarterly