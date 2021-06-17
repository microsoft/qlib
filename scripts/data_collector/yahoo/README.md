
- [Collector Data](#collector-data)
  - [Automatic update data](#automatic-update-of-daily-frequency-data(from-yahoo-finance))
  - [CN Data](#CN-Data)
    - [1d from yahoo](#1d-from-yahoocn)
    - [1d from qlib](#1d-from-qlibcn)
    - [using data(1d)](#using-data1d-cn)
    - [1min from yahoo](#1min-from-yahoocn)
    - [1min from qlib](#1min-from-qlibcn)
    - [using data(1min)](#using-data1min-cn)
  - [US Data](#CN-Data)
    - [1d from yahoo](#1d-from-yahoous)
    - [1d from qlib](#1d-from-qlibus)
    - [using data(1d)](#using-data1d-us)


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

### Automatic update of daily frequency data(from yahoo finance)
  > It is recommended that users update the data manually once (--trading_date 2021-05-25) and then set it to update automatically.

  * Automatic update of data to the "qlib" directory each trading day(Linux)
      * use *crontab*: `crontab -e`
      * set up timed tasks:

        ```
        * * * * 1-5 python <script path> update_data_to_bin --qlib_data_1d_dir <user data dir>
        ```
        * **script path**: *qlib/scripts/data_collector/yahoo/collector.py*

  * Manual update of data
      ```
      python qlib/scripts/data_collector/yahoo/collector.py update_data_to_bin --qlib_data_1d_dir <user data dir> --trading_date <start date> --end_date <end date>
      ```
      * *trading_date*: start of trading day
      * *end_date*: end of trading day(not included)

  * qlib/scripts/data_collector/yahoo/collector.py update_data_to_bin parameters:
      * *source_dir*: The directory where the raw data collected from the Internet is saved, default "Path(__file__).parent/source"
      * *normalize_dir*: Directory for normalize data, default "Path(__file__).parent/normalize"
      * *qlib_data_1d_dir*: the qlib data to be updated for yahoo, usually from: [download qlib data](https://github.com/microsoft/qlib/tree/main/scripts#download-cn-data)
      * *trading_date*: trading days to be updated, by default ``datetime.datetime.now().strftime("%Y-%m-%d")``
      * *end_date*: end datetime, default ``pd.Timestamp(trading_date + pd.Timedelta(days=1))``; open interval(excluding end)
      * *region*: region, value from ["CN", "US"], default "CN"


### CN Data

#### 1d from yahoo(CN)

```bash

# download from yahoo finance
python collector.py download_data --source_dir ~/.qlib/stock_data/source/cn_1d --region CN --start 2020-11-01 --end 2020-11-10 --delay 0.1 --interval 1d

# normalize
python collector.py normalize_data --source_dir ~/.qlib/stock_data/source/cn_1d --normalize_dir ~/.qlib/stock_data/source/cn_1d_nor --region CN --interval 1d

# dump data
cd qlib/scripts
python dump_bin.py dump_all --csv_path ~/.qlib/stock_data/source/cn_1d_nor --qlib_dir ~/.qlib/qlib_data/qlib_cn_1d --freq day --exclude_fields date,adjclose,dividends,splits,symbol

```

### 1d from qlib(CN)
```bash
python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/qlib_cn_1d --region cn
```

### using data(1d CN)

```python
import qlib
from qlib.data import D

qlib.init(provider_uri="~/.qlib/qlib_data/qlib_cn_1d", region="cn")
df = D.features(D.instruments("all"), ["$close"], freq="day")
```

#### 1min from yahoo(CN)

```bash

# download from yahoo finance
python collector.py download_data --source_dir ~/.qlib/stock_data/source/cn_1min --region CN --start 2020-11-01 --end 2020-11-10 --delay 0.1 --interval 1min

# normalize
python collector.py normalize_data --source_dir ~/.qlib/stock_data/source/cn_1min --normalize_dir ~/.qlib/stock_data/source/cn_1min_nor --region CN --interval 1min

# dump data
cd qlib/scripts
python dump_bin.py dump_all --csv_path ~/.qlib/stock_data/source/cn_1min_nor --qlib_dir ~/.qlib/qlib_data/qlib_cn_1min --freq 1min --exclude_fields date,adjclose,dividends,splits,symbol
```

### 1min from qlib(CN)
```bash
python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/qlib_cn_1min --interval 1min --region cn
```

### using data(1min CN)

```python
import qlib
from qlib.data import D

qlib.init(provider_uri="~/.qlib/qlib_data/qlib_cn_1min", region="cn")
df = D.features(D.instruments("all"), ["$close"], freq="1min")

```

### US Data

#### 1d from yahoo(US)

```bash

# download from yahoo finance
python collector.py download_data --source_dir ~/.qlib/stock_data/source/us_1d --region US --start 2020-11-01 --end 2020-11-10 --delay 0.1 --interval 1d

# normalize
python collector.py normalize_data --source_dir ~/.qlib/stock_data/source/us_1d --normalize_dir ~/.qlib/stock_data/source/us_1d_nor --region US --interval 1d

# dump data
cd qlib/scripts
python dump_bin.py dump_all --csv_path ~/.qlib/stock_data/source/us_1d_nor --qlib_dir ~/.qlib/stock_data/source/qlib_us_1d --freq day --exclude_fields date,adjclose,dividends,splits,symbol
```

#### 1d from qlib(US)

```bash
python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/qlib_us_1d --region us
```

### using data(1d US)

```python
# using
import qlib
from qlib.data import D

qlib.init(provider_uri="~/.qlib/qlib_data/qlib_us_1d", region="us")
df = D.features(D.instruments("all"), ["$close"], freq="day")

```


### Help
```bash
pythono collector.py collector_data --help
```

## Parameters

- interval: 1min or 1d
- region: CN or US
