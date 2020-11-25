.. _data:

================================
Data Layer: Data Framework & Usage
================================

Introduction
============================

``Data Layer`` provides user-friendly APIs to manage and retrieve data. It provides high-performance data infrastructure. 

It is designed for quantitative investment. For example, users could build formulaic alphas with ``Data Layer`` easily. Please refer to `Building Formulaic Alphas <../advanced/alpha.html>`_ for more details.

The introduction of ``Data Layer`` includes the following parts.

- Data Preparation
- Data API
- Data Loader
- Data Handler
- Dataset
- Cache
- Data and Cache File Structure


Data Preparation
============================

Qlib Format Data
------------------

We've specially designed a data structure to manage financial data, please refer to the `File storage design section in Qlib paper <https://arxiv.org/abs/2009.11189>`_ for detailed information.
Such data will be stored with filename suffix `.bin` (We'll call them `.bin` file, `.bin` format, or qlib format). `.bin` file is designed for scientific computing on finance data

Qlib Format Dataset
--------------------
``Qlib`` has provided an off-the-shelf dataset in `.bin` format, users could use the script ``scripts/get_data.py`` to download the dataset as follows.

.. code-block:: bash

    python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn

After running the above command, users can find china-stock data in Qlib format in the ``~/.qlib/csv_data/cn_data`` directory.

``Qlib`` also provides the scripts in ``scripts/data_collector`` to help users crawl the latest data on the Internet and convert it to qlib format.

When ``Qlib`` is initialized with this dataset, users could build and evaluate their own models with it.  Please refer to `Initialization <../start/initialization.html>`_ for more details.

Converting CSV Format into Qlib Format
-------------------------------------------

``Qlib`` has provided the script ``scripts/dump_bin.py`` to convert data in CSV format into `.bin` files (Qlib format).


Users can download the china-stock data in CSV format as follows for reference to the CSV format.

.. code-block:: bash

    python scripts/get_data.py csv_data_cn --target_dir ~/.qlib/csv_data/cn_data


Supposed that users prepare their CSV format data in the directory ``~/.qlib/csv_data/my_data``, they can run the following command to start the conversion.

.. code-block:: bash

    python scripts/dump_bin.py dump_all --csv_path  ~/.qlib/csv_data/my_data --qlib_dir ~/.qlib/qlib_data/my_data --include_fields open,close,high,low,volume,factor

After conversion, users can find their Qlib format data in the directory `~/.qlib/qlib_data/my_data`.

.. note::

    The arguments of `--include_fields` should correspond with the column names of CSV files. The columns names of dataset provided by ``Qlib`` should include open, close, high, low, volume and factor at least.
    
    - `open`
        The opening price
    - `close`
        The closing price
    - `high`
        The highest price
    - `low`
        The lowest price
    - `volume`
        The trading volume
    - `factor`
        The Restoration factor

    In the convention of `Qlib` data processing, `open, close, high, low, volume, money and factor` will be set to NaN if the stock is suspended. 

China-Stock Mode & US-Stock Mode
--------------------------------

- If users use ``Qlib`` in china-stock mode, china-stock data is required. Users can use ``Qlib`` in china-stock mode according to the following steps:
    - Download china-stock in qlib format, please refer to section `Qlib Format Dataset <#qlib-format-dataset>`_.
    - Initialize ``Qlib`` in china-stock mode
        Supposed that users download their Qlib format data in the directory ``~/.qlib/csv_data/cn_data``. Users only need to initialize ``Qlib`` as follows.
        
        .. code-block:: python

            from qlib.config import REG_CN
            qlib.init(provider_uri='~/.qlib/qlib_data/cn_data', region=REG_CN)
        

- If users use ``Qlib`` in US-stock mode, US-stock data is required. ``Qlib`` does not provide a script to download US-stock data. Users can use ``Qlib`` in US-stock mode according to the following steps:
    - Prepare data in CSV format
    - Convert data from CSV format to Qlib format,  please refer to section `Converting CSV Format into Qlib Format <#converting-csv-format-into-qlib-format>`_.
    - Initialize ``Qlib`` in US-stock mode
        Supposed that users prepare their Qlib format data in the directory ``~/.qlib/csv_data/us_data``. Users only need to initialize ``Qlib`` as follows.
        
        .. code-block:: python

            from qlib.config import REG_US
            qlib.init(provider_uri='~/.qlib/qlib_data/us_data', region=REG_US)
        

Data API
========================

Data Retrieval
---------------
Users can use APIs in ``qlib.data`` to retrieve data, please refer to `Data Retrieval <../start/getdata.html>`_.

Feature
------------------

``Qlib`` provides `Feature` and `ExpressionOps` to fetch the features according to users' needs.

- `Feature`
    Load data from the data provider. User can get the features like `$high`, `$low`, `$open`, `$close`, .etc, which should correspond with the arguments of `--include_fields`, please refer to section `Converting CSV Format into Qlib Format <#converting-csv-format-into-qlib-format>`_.

- `ExpressionOps`
    `ExpressionOps` will use operator for feature construction.
    To know more about  ``Operator``, please refer to `Operator API <../reference/api.html#module-qlib.data.ops>`_.

To know more about  ``Feature``, please refer to `Feature API <../reference/api.html#module-qlib.data.base>`_.

Filter
-------------------
``Qlib`` provides `NameDFilter` and `ExpressionDFilter` to filter the instruments according to users' needs.

- `NameDFilter`
    Name dynamic instrument filter. Filter the instruments based on a regulated name format. A name rule regular expression is required.

- `ExpressionDFilter`
    Expression dynamic instrument filter. Filter the instruments based on a certain expression. An expression rule indicating a certain feature field is required.
    
    - `basic features filter`: rule_expression = '$close/$open>5'
    - `cross-sectional features filter` : rule_expression = '$rank($close)<10'
    - `time-sequence features filter`: rule_expression = '$Ref($close, 3)>100'

To know more about ``Filter``, please refer to `Filter API <../reference/api.html#module-qlib.data.filter>`_.

Reference
-------------

To know more about ``Data API``, please refer to `Data API <../reference/api.html#data>`_.


Data Loader
=================

``Data Loader`` in ``Qlib`` is designed to load raw data from the original data source. It will be loaded and used in the ``Data Handler`` module.

The ``QlibDataLoader`` class in ``Qlib`` is such an interface that allows users to load raw data from the data source.

Interface
------------

Here are some interfaces of the ``QlibDataLoader`` class:

- `load(instruments, start_time=None, end_time=None)`
    - This method loads the data as pd.DataFrame
    - Parameters:
        - `instruments` : str or dict
            it can either be the market name or the config file of instruments generated by InstrumentProvider.
        - `start_time` : str
            start of the time range.
        - `end_time` : str
            end of the time range.
    - Returns:
        - The data being loaded with type `pd.DataFrame`

- `load_group_df(instruments, exprs: list, names: list, start_time=None, end_time=None)`
    -  This method loads the dataframe for specific group.
    - Parameters:
        - `instruments` : str or dict
            it can either be the market name or the config file of instruments generated by InstrumentProvider.
        - `exprs` : list
            the expressions to describe the content of the data.
        - `names` : list
            the name of the data.
        - `start_time` : str
            start of the time range.
        - `end_time` : str
            end of the time range.
    - Returns:
        - The queried data in type `pd.DataFrame`.

API
-----------

To know more about ``Data Loader``, please refer to `Data Loader API <../reference/api.html#module-qlib.data.dataset.loader>`_.


Data Handler
=================

The ``Data Handler`` module in ``Qlib`` is designed to handler those common data processing methods which will be used by most of the models.

Users can use ``Data Handler`` in an automatic workflow by ``qrun``, refer to `Workflow: Workflow Management <workflow.html>`_ for more details. 


Base Class & Interface
----------------------

In addition to use ``Data Handler`` in an automatic workflow with ``qrun``, ``Data Handler`` can be used as an independent module, by which users can easily preprocess data (standardization, remove NaN, etc.) and build datasets. 

In order to achieve so, ``Qlib`` provides a base class `qlib.data.dataset.DataHandlerLP <../reference/api.html#qlib.data.dataset.handler.DataHandlerLP>`_. The core idea of this class is that: we will have some leanable ``Processors`` which can learn the parameters of data processing. When new data comes in, these `trained` ``Processors`` can then infer on the new data and thus processing real-time data in an efficient way. More information about ``Processors`` will be listed in the next subsection.

Here are some important interfaces that ``DataHandlerLP`` provides:

- `__init__(instruments=None, start_time=None, end_time=None, data_loader: Tuple[dict, str, DataLoader] = None, infer_processors=[], learn_processors=[], process_type=PTYPE_A, **kwargs)`
    - Initialization of the class.
    - Parameters:
        - `infer_processors` : list
            - list of <description info> of processors to generate data for inference
            - example of <description info>:

            .. code-block::
            
                1) classname & kwargs:
                    {
                        "class": "MinMaxNorm",
                        "kwargs": {
                            "fit_start_time": "20080101",
                            "fit_end_time": "20121231"
                        }
                    }
                2) Only classname:
                    "DropnaFeature"
                3) object instance of Processor

        - `learn_processors` : list
            similar to infer_processors, but for generating data for learning models

        - `process_type`: str
            - PTYPE_I = 'independent'
                - self._infer will processed by infer_processors
                - self._learn will be processed by learn_processors
            - PTYPE_A = 'append'
                - self._infer will processed by infer_processors
                - self._learn will be processed by infer_processors + learn_processors
                    - (e.g. self._infer processed by learn_processors )

- `fetch(selector: Union[pd.Timestamp, slice, str] = slice(None, None), level: Union[str, int] = "datetime", col_set=DataHandler.CS_ALL, data_key: str = DK_I)`    
    - This method fetches data from underlying data source
    - Parameters:
        - `selector` : Union[pd.Timestamp, slice, str]
            describe how to select data by index.
        - `level` : Union[str, int]
            which index level to select the data.
        - `col_set` : str
            select a set of meaningful columns.(e.g. features, columns).
        - `data_key` : str
            The data to fetch:  DK_*.
    - Returns:
        - The retrieved results in the type: `pd.DataFrame`.

- `get_cols(col_set=DataHandler.CS_ALL, data_key: str = DK_I)`   
    - This method gets the column names.
    - Parameters:
        - `col_set` : str
            select a set of meaningful columns.(e.g. features, columns).
        - `data_key` : str
            the data to fetch:  DK_*.
    - Returns:
        - A list of column names.

If users want to load features and labels by config, users can inherit ``qlib.data.dataset.handler.ConfigDataHandler``, ``Qlib`` also provides some preprocess method in this subclass.
If users want to use qlib data, `QLibDataHandler` is recommended. Users can inherit their custom class from `QLibDataHandler`, which is also a subclass of `ConfigDataHandler`.


Processor
----------

The ``Processor`` module in ``Qlib`` is designed to be learnable and it is responsible for handling data processing such as `normalization` and `drop none/nan features/labels`.

``Qlib`` provides the following ``Processors``:

- ``DropnaProcessor``: `processor` that drops N/A features.
- ``DropnaLabel``: `processor` that drops N/A labels.
- ``TanhProcess``: `processor` that uses `tanh` to process noise data.
- ``ProcessInf``: `processor` that handles infinity values, it will be replaces by the mean of the column.
- ``Fillna``: `processor` that handles N/A values, which will fill the N/A value by 0 or other given number.
- ``MinMaxNorm``: `processor` that applies min-max normalization.
- ``ZscoreNorm``: `processor` that applies z-score normalization.
- ``CSZScoreNorm``: `processor` that applies cross sectional z-score normalization.
- ``CSRankNorm``: `processor` that applies cross sectional rank normalization.

Users can also create their own `processor` by inheriting the base class of ``Processor``. Please refer to the implementation of all the processors for more information (`Processor Link <https://github.com/microsoft/qlib/blob/main/qlib/data/dataset/processor.py>`_). 

API
---------

To know more about ``Processor``, please refer to `Processor API <../reference/api.html#module-qlib.data.dataset.processor>`_.


Usage
--------------

``Data Handler`` can be used as a single module, which provides the following mehtods:

- `get_split_data`
    - According to the start and end dates, return features and labels of the pandas DataFrame type used for the 'Model'

- `get_rolling_data`
    - According to the start and end dates, and `rolling_period`, an iterator is returned, which can be used to traverse the features and labels used for rolling.

Example
--------------

``Data Handler`` can be run with ``qrun`` by modifying the configuration file, and can also be used as a single module. 

Know more about how to run ``Data Handler`` with ``qrun``, please refer to `Workflow: Workflow Management <workflow.html>`_

Qlib provides implemented data handler `Alpha158`. The following example shows how to run `Alpha158` as a single module.

.. note:: Users need to initialize ``Qlib`` with `qlib.init` first, please refer to `initialization <../start/initialization.html>`_.


.. code-block:: Python

    import qlib
    from qlib.contrib.data.handler import Alpha158

    data_handler_config = {
        "start_time": "2008-01-01",
        "end_time": "2020-08-01",
        "fit_start_time": "2008-01-01",
        "fit_end_time": "2014-12-31",
        "instruments": "csi300",
    }

    if __name__ == "__main__":
        qlib.init()
        h = Alpha158(**data_handler_config)

        # get all the columns of the data
        print(h.get_cols())

        # fetch all the labels
        print(h.fetch(col_set="label"))

        # fetch all the features
        print(h.fetch(col_set="feature"))

API
---------

To know more about ``Data Handler``, please refer to `Data Handler API <../reference/api.html#module-qlib.data.dataset.handler>`_.


Dataset
=================

The ``Dataset`` module in ``Qlib`` aims to prepare data for model training and inferencing.

The motivation of this module is that we want to maximize the flexibility of of different models to handle data that are suitable for themselves. This module gives the model the rights to process their data in an unique way. For instance, models such as ``GBDT`` may work well on data that contains `nan` or `None` value, while neural networks such as ``DNN`` will break down on such data. 

The ``DatasetH`` class is the `dataset` with `Data Handler`. Here is the most important interface of the class:

- `prepare(segments: Union[List[str], Tuple[str], str, slice], col_set=DataHandler.CS_ALL, data_key=DataHandlerLP.DK_I, **kwargs)`
    - This method prepares the data for learning and inference.
    - Parameters:
        - `segments` : Union[List[str], Tuple[str], str, slice]
            Describe the scope of the data to be prepared
            Here are some examples:

            - 'train'

            - ['train', 'valid']

        - `col_set` : str
            The col_set will be passed to self._handler when fetching data.
        - `data_key` : str
            The data to fetch:  DK_*
            Default is DK_I, which indicate fetching data for **inference**.


API
---------

To know more about ``Dataset``, please refer to `Dataset API <../reference/api.html#module-qlib.data.dataset.__init__>`_.



Cache
==========

``Cache`` is an optional module that helps accelerate providing data by saving some frequently-used data as cache file. ``Qlib`` provides a `Memcache` class to cache the most-frequently-used data in memory, an inheritable `ExpressionCache` class, and an inheritable `DatasetCache` class.

Global Memory Cache
---------------------

`Memcache` is a global memory cache mechanism that composes of three `MemCacheUnit` instances to cache **Calendar**, **Instruments**, and **Features**. The `MemCache` is defined globally in `cache.py` as `H`. Users can use `H['c'], H['i'], H['f']` to get/set `memcache`.

.. autoclass:: qlib.data.cache.MemCacheUnit
    :members:

.. autoclass:: qlib.data.cache.MemCache
    :members:


ExpressionCache
-----------------

`ExpressionCache` is a cache mechanism that saves expressions such as **Mean($close, 5)**. Users can inherit this base class to define their own cache mechanism that saves expressions according to the following steps.

- Override `self._uri` method to define how the cache file path is generated
- Override `self._expression` method to define what data will be cached and how to cache it.

The following shows the details about the interfaces:

.. autoclass:: qlib.data.cache.ExpressionCache
    :members:

``Qlib`` has currently provided implemented disk cache `DiskExpressionCache` which inherits from `ExpressionCache` . The expressions data will be stored in the disk.

DatasetCache
-----------------

`DatasetCache` is a cache mechanism that saves datasets. A certain dataset is regulated by a stock pool configuration (or a series of instruments, though not recommended), a list of expressions or static feature fields, the start time, and end time for the collected features and the frequency. Users can inherit this base class to define their own cache mechanism that saves datasets according to the following steps.

- Override `self._uri` method to define how their cache file path is generated
- Override `self._expression` method to define what data will be cached and how to cache it.

The following shows the details about the interfaces:

.. autoclass:: qlib.data.cache.DatasetCache
    :members:

``Qlib`` has currently provided implemented disk cache `DiskDatasetCache` which inherits from `DatasetCache` . The datasets' data will be stored in the disk.



Data and Cache File Structure
==================================

We've specially designed a file structure to manage data and cache, please refer to the `File storage design section in Qlib paper <https://arxiv.org/abs/2009.11189>`_ for detailed information. The file structure of data and cache is listed as follows.

.. code-block:: json

    - data/
        [raw data] updated by data providers
        - calendars/
            - day.txt
        - instruments/
            - all.txt
            - csi500.txt
            - ...
        - features/
            - sh600000/
                - open.day.bin
                - close.day.bin
                - ...
            - ...
        [cached data] updated when raw data is updated
        - calculated features/
            - sh600000/
                - [hash(instrtument, field_expression, freq)]
                    - all-time expression -cache data file
                    - .meta : an assorted meta file recording the instrument name, field name, freq, and visit times
            - ...
        - cache/
            - [hash(stockpool_config, field_expression_list, freq)]
                - all-time Dataset-cache data file
                - .meta : an assorted meta file recording the stockpool config, field names and visit times
                - .index : an assorted index file recording the line index of all calendars
            - ...

