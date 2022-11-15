# Global Investment Opportunity with Intelligent Asset Allocator (with Qlib)

## Install Qlib

- Run below command to install `qlib from local src`

  ```
  pip install ../../
  ```
#####  or
- Run below command to install `qlib from repository`

  ```
  pip install pyqlib
  ```

## Data Preparation

- Run below command to `download and prepare the data from Yahoo Finance using data collector`

  ```
python ../../scripts/data_collector/yahoo/collector.py update_data_to_bin --qlib_data_1d_dir ~/.qlib/qlib_data/us_data --trading_date 2022-10-31 --region US
  ```