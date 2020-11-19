# Download Qlib Data


## Download CN Data

```bash
python get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn
```

## Downlaod US Data

> The US stock code contains 'PRN', and the directory cannot be created on Windows system

```bash
python get_data.py qlib_data --target_dir ~/.qlib/qlib_data/us_data --region us
```

## Download CN Simple Data

```bash
python get_data.py qlib_data --name qlib_data_simple --target_dir ~/.qlib/qlib_data/cn_data --region cn
```

## Help

```bash
python get_data.py qlib_data --help
```

