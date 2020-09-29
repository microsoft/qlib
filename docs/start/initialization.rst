.. _initialization:
====================
Qlib Initialization
====================

.. currentmodule:: qlib


Initialization
=========================

Please follow the steps below to initialize ``Qlib``.

- Download and prepare the Data: execute the following command to download stock data. Please pay `attention` that the data is collected from `Yahoo Finance <https://finance.yahoo.com/lookup>`_ and the data might not be perfect. We recommend users to prepare their own data if they have high-quality datasets. Please refer to `Data  <../component/data.html#converting-csv-format-into-qlib-format>` for more information about customized dataset.
    .. code-block:: bash
    
        python scripts/get_data.py qlib_data_cn --target_dir ~/.qlib/qlib_data/cn_data
    Please refer to `Data Preparation  <../component/data.html#data-preparation>`_ for more information about `get_data.py`,


- Initialize Qlib before calling other APIs: run following code in python.

    .. code-block:: Python

        import qlib
        # region in [REG_CN, REG_US]
        from qlib.config import REG_CN
        provider_uri = "~/.qlib/qlib_data/cn_data"  # target_dir
        qlib.init(provider_uri=provider_uri, region=REG_CN)
    
.. note::
   Do not import qlib package in the repository directory  of ``Qlib``, otherwise, errors may occur.

Parameters
-------------------

Besides `provider_uri` and `region`, `qlib.init` has other parameters. The following are several important parameters of `qlib.init`:

- `provider_uri`
    Type: str. The URI of the Qlib data. For example, it could be the location where the data loaded by ``get_data.py`` are stored.
- `region`
    Type: str, optional parameter(default: `qlib.config.REG_CN`).
        Currently: ``qlib.config.REG_US`` ('us') and ``qlib.config.REG_CN`` ('cn') is supported. Different value of  `region` will result in different stock market mode.
        - ``qlib.config.REG_US``: US stock market.
        - ``qlib.config.REG_CN``: China stock market.

        Different modse will result in different trading limitations and costs.
- `redis_host`
    Type: str, optional parameter(default: "127.0.0.1"), host of `redis`
        The lock and cache mechanism relies on redis.
- `redis_port`
    Type: int, optional parameter(default: 6379), port of `redis`

    .. note:: 
        
        The value of `region` should be aligned with the data stored in `provider_uri`. Currently, ``scripts/get_data.py`` only provides China stock market data. If users want to use the US stock market data, they should prepare their own US-stock data in `provider_uri` and switch to US-stock mode.

    .. note::
        
        If Qlib fails to connect redis via `redis_host` and `redis_port`, cache mechanism will not be used! Please refer to `Cache <../component/data.html#cache>`_ for details.
