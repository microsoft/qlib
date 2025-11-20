
- [Download Qlib Data](#Download-Qlib-Data)
  - [Download CN Data](#Download-CN-Data)
  - [Download US Data](#Download-US-Data)
  - [Download CN Simple Data](#Download-CN-Simple-Data)
  - [Help](#Help)
- [Using in Qlib](#Using-in-Qlib)
  - [US data](#US-data)
  - [CN data](#CN-data)


## Download Qlib Data


### Download CN Data

```bash
# daily data
python get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn

# 1min  data (Optional for running non-high-frequency strategies)
python get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data_1min --region cn --interval 1min
```

### Download US Data


```bash
python get_data.py qlib_data --target_dir ~/.qlib/qlib_data/us_data --region us
```

### Download CN Simple Data

```bash
python get_data.py qlib_data --name qlib_data_simple --target_dir ~/.qlib/qlib_data/cn_data --region cn
```

### Help

```bash
python get_data.py qlib_data --help
```

## Using in Qlib
> For more information: https://qlib.readthedocs.io/en/latest/start/initialization.html


### US data

> Need to download data first: [Download US Data](#Download-US-Data)

```python
import qlib
from qlib.config import REG_US
provider_uri = "~/.qlib/qlib_data/us_data"  # target_dir
qlib.init(provider_uri=provider_uri, region=REG_US)
```

### CN data

> Need to download data first: [Download CN Data](#Download-CN-Data)

```python
import qlib
from qlib.constant import REG_CN

provider_uri = "~/.qlib/qlib_data/cn_data"  # target_dir
qlib.init(provider_uri=provider_uri, region=REG_CN)
```

## Use Crowd Sourced Data
The is also a [crowd sourced version of qlib data](data_collector/crowd_source/README.md): https://github.com/chenditc/investment_data/releases
```bash
wget https://github.com/chenditc/investment_data/releases/latest/download/qlib_bin.tar.gz
tar -zxvf qlib_bin.tar.gz -C ~/.qlib/qlib_data/cn_data --strip-components=2
```
