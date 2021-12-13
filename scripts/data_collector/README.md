# Data Collector

## Introduction

Scripts for data collection

- yahoo: get *US/CN* stock data from *Yahoo Finance*
- fund: get fund data from *http://fund.eastmoney.com*
- cn_index: get *CN index* from *http://www.csindex.com.cn*, *CSI300*/*CSI100*
- us_index: get *US index* from *https://en.wikipedia.org/wiki*, *SP500*/*NASDAQ100*/*DJIA*/*SP400*
- contrib: scripts for some auxiliary functions


## Custom Data Collection

> Specific implementation reference: https://github.com/microsoft/qlib/tree/main/scripts/data_collector/yahoo

1. Create a dataset code directory in the current directory
2. Add `collector.py`
   - add collector class:
     ```python
     CUR_DIR = Path(__file__).resolve().parent
     sys.path.append(str(CUR_DIR.parent.parent))
     from data_collector.base import BaseCollector, BaseNormalize, BaseRun
     class UserCollector(BaseCollector):
         ...
     ```
   - add normalize class:
     ```python
     class UserNormalzie(BaseNormalize):
         ...
     ```
   - add `CLI` class:
     ```python
     class Run(BaseRun):
         ...
     ```
3. add `README.md`
4. add `requirements.txt`


## Description of dataset

  |             | Basic data                                                                                                       |
  |------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------|
  | Features    | **Price/Volume**: <br>&nbsp;&nbsp; - $close/$open/$low/$high/$volume/$change/$factor                             |
  | Calendar    | **\<freq>.txt**: <br>&nbsp;&nbsp; - day.txt<br>&nbsp;&nbsp;  - 1min.txt                                          |
  | Instruments | **\<market>.txt**: <br>&nbsp;&nbsp; - required: **all.txt**; <br>&nbsp;&nbsp;  - csi300.txt/csi500.txt/sp500.txt |

  - `Features`: data, **digital**
    - if not **adjusted**, **factor=1**

### Data-dependent component

> To make the component running correctly, the dependent data are required

  | Component      | required data                                     |
  |---------------------------------------------------|--------------------------------|
  | Data retrieval | Features, Calendar, Instrument                    |
  | Backtest       | **Features[Price/Volume]**, Calendar, Instruments |