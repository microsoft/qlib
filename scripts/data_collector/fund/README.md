# Collect Fund Data

> *Please pay **ATTENTION** that the data is collected from [天天基金网](https://fund.eastmoney.com/) and the data might not be perfect. We recommend users to prepare their own data if they have high-quality dataset. For more information, users can refer to the [related document](https://qlib.readthedocs.io/en/latest/component/data.html#converting-csv-format-into-qlib-format)*

## Requirements

```bash
pip install -r requirements.txt
```

## Collector Data


### CN Data

#### 1d from East Money

```bash

# download from eastmoney.com
python collector.py download_data --source_dir ~/.qlib/fund_data/source/cn_1d --region CN --start 2020-11-01 --end 2020-11-10 --delay 0.1 --interval 1d


# dump data
cd qlib/scripts
python dump_bin.py dump_all --csv_path ~/.qlib/stock_data/source/cn_1d_nor --qlib_dir ~/.qlib/qlib_data/qlib_cn_1d --freq day --exclude_fields date,adjclose,dividends,splits,symbol

```

### using data

```python
import qlib
from qlib.data import D

qlib.init(provider_uri="~/.qlib/qlib_data/qlib_cn_1d", region="CN")
df = D.features(D.instruments("all"), ["$close"], freq="day")
```


### Help
```bash
pythono collector.py collector_data --help
```

## Parameters

- interval: 1min or 1d
- region: CN or US
