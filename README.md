[![Python Versions](https://img.shields.io/pypi/pyversions/pyqlib.svg?logo=python&logoColor=white)](https://pypi.org/project/pyqlib/#files)
[![Platform](https://img.shields.io/badge/platform-linux%20%7C%20windows%20%7C%20macos-lightgrey)](https://pypi.org/project/pyqlib/#files)
[![PypI Versions](https://img.shields.io/pypi/v/pyqlib)](https://pypi.org/project/pyqlib/#files)
[![Documentation Status](https://readthedocs.org/projects/qlib/badge/?version=latest)](https://qlib.readthedocs.io/en/latest/?badge=latest)
![Upload Python Package](https://github.com/microsoft/qlib/workflows/Upload%20Python%20Package/badge.svg)
[![License](https://img.shields.io/pypi/l/pyqlib)](LICENSE)
[![Join the chat at https://gitter.im/Microsoft/qlib](https://badges.gitter.im/Microsoft/qlib.svg)](https://gitter.im/Microsoft/qlib?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)


<p align="center">
  <img src="http://fintech.msra.cn/images/logo/1.png" />
</p>


Qlib is an AI-oriented quantitative investment platform, which aims to realize the potential, empower the research, and create the value of AI technologies in quantitative investment.

With Qlib, you can easily try your ideas to create better Quant investment strategies.

For more details, please refer to our paper ["Qlib: An AI-oriented Quantitative Investment Platform"](https://arxiv.org/abs/2009.11189).

- [Framework of Qlib](#framework-of-qlib)
- [Quick Start](#quick-start)
  - [Installation](#installation)
  - [Data Preparation](#data-preparation)
  - [Auto Quant Research Workflow](#auto-quant-research-workflow)
  - [Building Customized Quant Research Workflow by Code](#building-customized-quant-research-workflow-by-code)
- [Quant Model Zoo](#quant-model-zoo)
- [Quant Dataset Zoo](#quant-dataset-zoo)
- [More About Qlib](#more-about-qlib)
- [Offline Mode and Online Mode](#offline-mode-and-online-mode)
  - [Performance of Qlib Data Server](#performance-of-qlib-data-server)
- [Contributing](#contributing)



# Framework of Qlib

<div style="align: center">
<img src="http://fintech.msra.cn/images/framework.png" />
</div>


At the module level, Qlib is a platform that consists of the above components. The components are designed as loose-coupled modules and each component could be used stand-alone.

| Name                | Description                                                                                                                                                                                                                                                   |
| ------              | -----                                                                                                                                                                                                                                                         |
| `Data layer`        | `DataServer` focuses on providing high-performance infrastructure for users to manage and retrieve raw data. `DataEnhancement` will preprocess the data and provide the best dataset to be fed into the models.                                                    |
| `Interday Model`    | `Interday model` focuses on producing prediction scores (aka. _alpha_). Models are trained by `Model Creator` and managed by `Model Manager`. Users could choose one or multiple models for prediction. Multiple models could be combined with `Ensemble` module. |
| `Interday Strategy` | `Portfolio Generator` will take prediction scores as input and output the orders based on the current position to achieve the target portfolio.                                                                                                                      |
| `Intraday Trading`  | `Order Executor` is responsible for executing orders output by `Interday Strategy` and returning the executed results.                                                                                                                                        |
| `Analysis`          | Users could get a detailed analysis report of forecasting signals and portfolios in this part.                                                                                                                                                                     |

* The modules with hand-drawn style are under development and will be released in the future.
* The modules with dashed borders are highly user-customizable and extendible.


# Quick Start

This quick start guide tries to demonstrate
1. It's very easy to build a complete Quant research workflow and try your ideas with _Qlib_.
1. Though with *public data* and *simple models*, machine learning technologies **work very well** in practical Quant investment.

## Installation

Users can easily install ``Qlib`` by pip according to the following command

```bash
  pip install pyqlib
```

Also, Users can install ``Qlib`` by the source code according to the following steps:

* Before installing ``Qlib`` from source, users need to install some dependencies:

  ```bash
  pip install numpy
  pip install --upgrade  cython
  ```

* Clone the repository and install ``Qlib``:

  ```bash
  git clone https://github.com/microsoft/qlib.git && cd qlib
  python setup.py install
  ```


## Data Preparation
Load and prepare data by running the following code:
  ```bash
  python scripts/get_data.py qlib_data_cn --target_dir ~/.qlib/qlib_data/cn_data
  ```

This dataset is created by public data collected by [crawler scripts](scripts/data_collector/), which have been released in
the same repository.
Users could create the same dataset with it. 

*Please pay **ATTENTION** that the data is collected from [Yahoo Finance](https://finance.yahoo.com/lookup) and the data might not be perfect. We recommend users to prepare their own data if they have high-quality dataset. For more information, users can refer to the [related document](https://qlib.readthedocs.io/en/latest/component/data.html#converting-csv-format-into-qlib-format)*.

<!-- 
- Run the initialization code and get stock data:

  ```python
  import qlib
  from qlib.data import D
  from qlib.config import REG_CN

  # Initialization
  mount_path = "~/.qlib/qlib_data/cn_data"  # target_dir
  qlib.init(mount_path=mount_path, region=REG_CN)

  # Get stock data by Qlib
  # Load trading calendar with the given time range and frequency
  print(D.calendar(start_time='2010-01-01', end_time='2017-12-31', freq='day')[:2])

  # Parse a given market name into a stockpool config
  instruments = D.instruments('csi500')
  print(D.list_instruments(instruments=instruments, start_time='2010-01-01', end_time='2017-12-31', as_list=True)[:6])

  # Load features of certain instruments in given time range
  instruments = ['SH600000']
  fields = ['$close', '$volume', 'Ref($close, 1)', 'Mean($close, 3)', '$high-$low']
  print(D.features(instruments, fields, start_time='2010-01-01', end_time='2017-12-31', freq='day').head())
  ```
 -->

## Auto Quant Research Workflow
Qlib provides a tool named `Estimator` to run the whole workflow automatically (including building dataset, training models, backtest and evaluation). You can start an auto quant research workflow and have a graphical reports analysis according to the following steps: 

1. Quant Research Workflow: Run  `Estimator` with [estimator_config.yaml](examples/estimator/estimator_config.yaml) as following. (*Please note that this may **not work** under MacOS with Python 3.8 due to the incompatibility of the `sacred` package we use with Python 3.8. We will fix this bug in the future.*)
    ```bash
      cd examples  # Avoid running program under the directory contains `qlib`
      estimator -c estimator/estimator_config.yaml
    ```
    The result of `Estimator` is as follows, please refer to please refer to [Intraday Trading](https://qlib.readthedocs.io/en/latest/component/backtest.html) for more details about the result. 

    ```bash

                                                      risk
    excess_return_without_cost mean               0.000675
                               std                0.005456
                               annualized_return  0.170077
                               information_ratio  1.963824
                               max_drawdown      -0.063646
    excess_return_with_cost    mean               0.000479
                               std                0.005453
                               annualized_return  0.120776
                               information_ratio  1.395116
                               max_drawdown      -0.071216



    ```
    Here are detailed documents for [Estimator](https://qlib.readthedocs.io/en/latest/component/estimator.html).

2. Graphical Reports Analysis: Run `examples/estimator/analyze_from_estimator.ipynb` with `jupyter notebook` to get graphical reports
    - Forecasting signal (model prediction) analysis
      - Cumulative Return of groups
      ![Cumulative Return](http://fintech.msra.cn/images/analysis/analysis_model_cumulative_return.png?v=0.1)
      - Return distribution
      ![long_short](http://fintech.msra.cn/images/analysis/analysis_model_long_short.png?v=0.1)
      - Information Coefficient (IC)
      ![Information Coefficient](http://fintech.msra.cn/images/analysis/analysis_model_IC.png?v=0.1)        
      ![Monthly IC](http://fintech.msra.cn/images/analysis/analysis_model_monthly_IC.png?v=0.1)
      ![IC](http://fintech.msra.cn/images/analysis/analysis_model_NDQ.png?v=0.1)
      - Auto Correlation of forecasting signal (model prediction)
      ![Auto Correlation](http://fintech.msra.cn/images/analysis/analysis_model_auto_correlation.png?v=0.1)

    - Portfolio analysis
      - Backtest return
      ![Report](http://fintech.msra.cn/images/analysis/report.png?v=0.1)
      <!-- 
      - Score IC
      ![Score IC](docs/_static/img/score_ic.png)
      - Cumulative Return
      ![Cumulative Return](docs/_static/img/cumulative_return.png)
      - Risk Analysis
      ![Risk Analysis](docs/_static/img/risk_analysis.png)
      - Rank Label
      ![Rank Label](docs/_static/img/rank_label.png)
      -->

## Building Customized Quant Research Workflow by Code
The automatic workflow may not suite the research workflow of all Quant researchers. To support a flexible Quant research workflow, Qlib also provides a modularized interface to allow researchers to build their own workflow by code. [Here](examples/train_backtest_analyze.ipynb) is a demo for customized Quant research workflow by code


# Quant Model Zoo

Here is a list of models build on `Qlib`.
- [GBDT based on lightgbm](qlib/contrib/model/gbdt.py)
- [MLP based on pytroch](qlib/contrib/model/pytorch_nn.py)

Your PR of new Quant models is highly welcomed.

# Quant Dataset Zoo
Dataset plays a very important role in Quant. Here is a list of the datasets build on `Qlib`.
- [Alpha360](./qlib/contrib/estimator/handler.py)
- [Alpha158](./qlib/contrib/estimator/handler.py)

Here is a tutorial to build dataset with `Qlib`.
Your PR to build new Quant dataset is highly welcomed.

# More About Qlib
The detailed documents are organized in [docs](docs/).
[Sphinx](http://www.sphinx-doc.org) and the readthedocs theme is required to build the documentation in html formats. 
```bash
cd docs/
conda install sphinx sphinx_rtd_theme -y
# Otherwise, you can install them with pip
# pip install sphinx sphinx_rtd_theme
make html
```
You can also view the [latest document](http://qlib.readthedocs.io/) online directly.

Qlib is in active and continuing development. Our plan is in the roadmap, which is managed as a [github project](https://github.com/microsoft/qlib/projects/1).



# Offline Mode and Online Mode
The data server of Qlib can either deployed as `Offline` mode or `Online` mode. The default mode is offline mode.

Under `Offline` mode, the data will be deployed locally. 

Under `Online` mode, the data will be deployed as a shared data service. The data and their cache will be shared by all the clients. The data retrieval performance is expected to be improved due to a higher rate of cache hits. It will consume less disk space, too. The documents of the online mode can be found in [Qlib-Server](https://qlib-server.readthedocs.io/). The online mode can be deployed automatically with [Azure CLI based scripts](https://qlib-server.readthedocs.io/en/latest/build.html#one-click-deployment-in-azure). The source code of online data server can be found in [Qlib-Server repository](https://github.com/microsoft/qlib-server).

## Performance of Qlib Data Server
The performance of data processing is important to data-driven methods like AI technologies. As an AI-oriented platform, Qlib provides a solution for data storage and data processing. To demonstrate the performance of Qlib data server, we
compare it with several other data storage solutions. 

We evaluate the performance of several storage solutions by finishing the same task,
which creates a dataset (14 features/factors) from the basic OHLCV daily data of a stock market (800 stocks each day from 2007 to 2020). The task involves data queries and processing.

|                         | HDF5      | MySQL     | MongoDB   | InfluxDB  | Qlib -E -D  | Qlib +E -D   | Qlib +E +D  |
| --                      | ------    | ------    | --------  | --------- | ----------- | ------------ | ----------- |
| Total (1CPU) (seconds)  | 184.4±3.7 | 365.3±7.5 | 253.6±6.7 | 368.2±3.6 | 147.0±8.8   | 47.6±1.0     | **7.4±0.3** |
| Total (64CPU) (seconds) |           |           |           |           | 8.8±0.6     | **4.2±0.2**  |             |
* `+(-)E` indicates with (out) `ExpressionCache`
* `+(-)D` indicates with (out) `DatasetCache`

Most general-purpose databases take too much time on loading data. After looking into the underlying implementation, we find that data go through too many layers of interfaces and unnecessary format transformations in general-purpose database solutions.
Such overheads greatly slow down the data loading process.
Qlib data are stored in a compact format, which is efficient to be combined into arrays for scientific computation.





# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the right to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
