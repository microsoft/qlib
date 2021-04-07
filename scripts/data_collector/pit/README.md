# Collect Point-in-Time Data

> *Please pay **ATTENTION** that the data is collected from [baostock](http://baostock.com) and the data might not be perfect. We recommend users to prepare their own data if they have high-quality dataset. For more information, users can refer to the [related document](https://qlib.readthedocs.io/en/latest/component/data.html#converting-csv-format-into-qlib-format)*

## Requirements

```bash
pip install -r requirements.txt
```

## Collector Data


### Download Quarterly CN Data

#### 1d from East Money

```bash

# download from baostock.com
python collector.py download_data --source_dir ~/.qlib/cn_data/source/pit_quarter --start 2010-01-01 --end 2021-01-01 --interval quarterly

```
