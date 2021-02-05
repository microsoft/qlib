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


### CN Data

#### 1d from yahoo

```bash

# download from yahoo finance
python collector.py download_data --source_dir ~/.qlib/stock_data/source/cn_1d --region CN --start 2020-11-01 --end 2020-11-10 --delay 0.1 --interval 1d

# normalize
python collector.py normalize_data --source_dir ~/.qlib/stock_data/source/cn_1d --normalize_dir ~/.qlib/stock_data/source/cn_1d_nor --region CN --interval 1d

# dump data
cd qlib/scripts
python dump_bin.py dump_all --csv_path ~/.qlib/stock_data/source/cn_1d_nor --qlib_dir ~/.qlib/qlib_data/qlib_cn_1d --freq day --exclude_fields date,adjclose,dividends,splits,symbol

```

### 1d from qlib
```bash
python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/qlib_cn_1d --region cn
```

### using data

```python
import qlib
from qlib.data import D

qlib.init(provider_uri="~/.qlib/qlib_data/qlib_cn_1d", region="CN")
df = D.features(D.instruments("all"), ["$close"], freq="day")
```

#### 1min from yahoo

```bash

# download from yahoo finance
python collector.py download_data --source_dir ~/.qlib/stock_data/source/cn_1min --region CN --start 2020-11-01 --end 2020-11-10 --delay 0.1 --interval 1min

# normalize
python collector.py normalize_data --source_dir ~/.qlib/stock_data/source/cn_1min --normalize_dir ~/.qlib/stock_data/source/cn_1min_nor --region CN --interval 1min

# dump data
cd qlib/scripts
python dump_bin.py dump_all --csv_path ~/.qlib/stock_data/source/cn_1min_nor --qlib_dir ~/.qlib/qlib_data/qlib_cn_1min --freq 1min --exclude_fields date,adjclose,dividends,splits,symbol
```

### 1min from qlib
```bash
python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/qlib_cn_1min --interval 1min --region cn
```

### using data

```python
import qlib
from qlib.data import D

qlib.init(provider_uri="~/.qlib/qlib_data/qlib_cn_1min", region="CN")
df = D.features(D.instruments("all"), ["$close"], freq="1min")

```

### US Data

#### 1d from yahoo

```bash

# download from yahoo finance
python collector.py download_data --source_dir ~/.qlib/stock_data/source/us_1d --region US --start 2020-11-01 --end 2020-11-10 --delay 0.1 --interval 1d

# normalize
python collector.py normalize_data --source_dir ~/.qlib/stock_data/source/us_1d --normalize_dir ~/.qlib/stock_data/source/us_1d_nor --region US --interval 1d

# dump data
cd qlib/scripts
python dump_bin.py dump_all --csv_path ~/.qlib/stock_data/source/cn_1d_nor --qlib_dir ~/.qlib/stock_data/source/qlib_us_1d --freq day --exclude_fields date,adjclose,dividends,splits,symbol
```

#### 1d from qlib

```bash
python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/qlib_us_1d --region us
```

### using data

```python
# using
import qlib
from qlib.data import D

qlib.init(provider_uri="~/.qlib/qlib_data/qlib_us_1d", region="US")
df = D.features(D.instruments("all"), ["$close"], freq="day")

```


### Help
```bash
pythono collector.py collector_data --help
```

## Parameters

- interval: 1min or 1d
- region: CN or US
