.. _initialization:
====================
Qlib Initialization
====================

.. currentmodule:: qlib


Initialization
=========================

Please execute the following process to initialize ``Qlib``.

- Download and prepare the Data: execute the following command to download the stock data.
    .. code-block:: bash
    
        python scripts/get_data.py qlib_data_cn --target_dir ~/.qlib/qlib_data/cn_data

    Know more about how to use ``get_data.py``, refer to `Raw Data  <../advanced/data.html#raw-data>`_.


- Run the initialization code: run the following code in python:

    .. code-block:: Python

        import qlib
        # region in [REG_CN, REG_US]
        from qlib.config import REG_CN
        provider_uri = "~/.qlib/qlib_data/cn_data"  # target_dir
        qlib.init(provider_uri=provider_uri, region=REG_CN)
    


Parameters
-------------------

In fact, in addition to `provider_uri` and `region`, `qlib.init` has other parameters. The following are all the parameters of `qlib.init`:

- `provider_uri`
    Type: str. The local directory where the data loaded by ``get_data.py`` is stored.
- `region`
    Type: str, optional parameter(default: ``qlib.config.REG_CN``).
        Currently: ``qlib.config.REG_US``('us') and ``qlib.config.REG_CN``('cn') is supported. Different value of  ``region`` will
        result in different stock market mode.

        - ``qlib.config.REG_US``: US stock market.
        - ``qlib.config.REG_CN``: China stock market.
- `redis_host`
    Type: str, optional parameter(default: "127.0.0.1"), host of `redis`
        The lock and cache mechanism relies on redis.
- `redis_port`
    Type: int, optional parameter(default: 6379), port of `redis`

    .. note:: 
        
        The value of `region` should be aligned with the data stored in `provider_uri`. Currently, ``scripts/get_data.py`` only provides China stock market data. If users want to use the US stock market data, they should prepare their own US-stock data in `provider_uri` and switch to US-stock mode.

    .. note::
        
        If redis connection failed with `redis_host` and `redis_port`, cache will not be used! Please refer to `Cache <../advanced/cache.rst>`_.
