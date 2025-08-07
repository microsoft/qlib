## Collector Data

### Get Qlib data(`bin file`)

  - get data: `python scripts/get_data.py qlib_data`
  - parameters:
    - `target_dir`: save dir, by default *~/.qlib/qlib_data/cn_data_5min*
    - `version`: dataset version, value from [`v2`], by default `v2`
      - `v2` end date is *2022-12*
    - `interval`: `5min`
    - `region`: `hs300`
    - `delete_old`: delete existing data from `target_dir`(*features, calendars, instruments, dataset_cache, features_cache*), value from [`True`, `False`], by default `True`
    - `exists_skip`: traget_dir data already exists, skip `get_data`, value from [`True`, `False`], by default `False`
  - examples:
    ```bash
    # hs300 5min
    python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/hs300_data_5min --region hs300 --interval 5min
    ```
    
### Collector *Baostock high frequency* data to qlib
> collector *Baostock high frequency* data and *dump* into `qlib` format.
> If the above ready-made data can't meet users' requirements,  users can follow this section to crawl the latest data and convert it to qlib-data.
  1. download data to csv: `python scripts/data_collector/baostock_5min/collector.py download_data`
     
     This will download the raw data such as date, symbol, open, high, low, close, volume, amount, adjustflag from baostock to a local directory. One file per symbol.
     - parameters:
          - `source_dir`: save the directory
          - `interval`: `5min`
          - `region`: `HS300`
          - `start`: start datetime, by default *None*
          - `end`: end datetime, by default *None*
     - examples:
          ```bash
          # cn 5min data
          python collector.py download_data --source_dir ~/.qlib/stock_data/source/hs300_5min_original --start 2022-01-01 --end 2022-01-30 --interval 5min --region HS300
          ```
  2. normalize data: `python scripts/data_collector/baostock_5min/collector.py normalize_data`
     
     This will:
     1. Normalize high, low, close, open price using adjclose.
     2. Normalize the high, low, close, open price so that the first valid trading date's close price is 1. 
     - parameters:
          - `source_dir`: csv directory
          - `normalize_dir`: result directory
          - `interval`: `5min`
            > if **`interval == 5min`**, `qlib_data_1d_dir` cannot be `None`
          - `region`: `HS300`
          - `date_field_name`: column *name* identifying time in csv files, by default `date`
          - `symbol_field_name`: column *name* identifying symbol in csv files, by default `symbol`
          - `end_date`: if not `None`, normalize the last date saved (*including end_date*); if `None`, it will ignore this parameter; by default `None`
          - `qlib_data_1d_dir`: qlib directory(1d data)
            if interval==5min, qlib_data_1d_dir cannot be None, normalize 5min needs to use 1d data;
            ```
                # qlib_data_1d can be obtained like this:
                python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data --interval 1d --region cn --version v3
            ```
      - examples:
        ```bash
        # normalize 5min cn
        python collector.py normalize_data --qlib_data_1d_dir ~/.qlib/qlib_data/cn_data --source_dir ~/.qlib/stock_data/source/hs300_5min_original --normalize_dir ~/.qlib/stock_data/source/hs300_5min_nor --region HS300 --interval 5min
        ```
  3. dump data: `python scripts/dump_bin.py dump_all`
    
     This will convert the normalized csv in `feature` directory as numpy array and store the normalized data one file per column and one symbol per directory. 
    
     - parameters:
       - `data_path`: stock data path or directory, **normalize result(normalize_dir)**
       - `qlib_dir`: qlib(dump) data director
       - `freq`: transaction frequency, by default `day`
         > `freq_map = {1d:day, 5mih: 5min}`
       - `max_workers`: number of threads, by default *16*
       - `include_fields`: dump fields, by default `""`
       - `exclude_fields`: fields not dumped, by default `"""
         > dump_fields = `include_fields if include_fields else set(symbol_df.columns) - set(exclude_fields) exclude_fields else symbol_df.columns`
       - `symbol_field_name`: column *name* identifying symbol in csv files, by default `symbol`
       - `date_field_name`: column *name* identifying time in csv files, by default `date`
       - `file_suffix`: stock data file format, by default ".csv"
     - examples:
       ```bash
       # dump 5min cn
       python dump_bin.py dump_all --data_path ~/.qlib/stock_data/source/hs300_5min_nor --qlib_dir ~/.qlib/qlib_data/hs300_5min_bin --freq 5min --exclude_fields date,symbol
       ```