.. _cache:
====================
Cache: Frequently-Used Data
====================

.. currentmodule:: qlib

The `cache` is a pluggable module to help accelerate providing data by saving some frequently-used data as cache file. Qlib provides a `Memcache` class to cache the most-frequently-used data in memory, an inheritable `ExpressionCache` class, and an inheritable `DatasetCache` class.

`Memcache` is a memory cache mechanism that composes of three `MemCacheUnit` instances to cache **Calendar**, **Instruments**, and **Features**. The MemCache is defined globally in `cache.py` as `H`. User can use `H['c'], H['i'], H['f']` to get/set memcache.

.. autoclass:: qlib.data.cache.MemCacheUnit
    :members:

.. autoclass:: qlib.data.cache.MemCache
    :members:

`ExpressionCache` is a disk cache mechanism that saves expressions such as **Mean($close, 5)**. Users can inherit this base class to define their own cache mechanism. Users need to override `self._uri` method to define how their cache file path is generated, `self._expression` method to define what data they want to cache and how to cache it.

`DatasetCache` is a disk cache mechanism that saves datasets. A certain dataset is regulated by a stockpool configuration (or a series of instruments, though not recommended), a list of expressions or static feature fields, the start time and end time for the collected features and the frequency. Users need to override `self._uri` method to define how their cache file path is generated, `self._expression` method to define what data they want to cache and how to cache it.

`ExpressionCache` and `DatasetCache` actually provides the same interfaces with `ExpressionProvider` and `DatasetProvider` so that the disk cache layer is transparent to users and will only be used if they want to define their own cache mechanism. The users can plug the cache mechanism into the server system by assigning the cache class they want to use in `config.py`:

.. code-block:: python

    'ExpressionCache': 'ServerExpressionCache',
    'DatasetCache': 'ServerDatasetCache',

User can find the cache interface here.

ExpressionCache
====================
.. autoclass:: qlib.data.cache.ExpressionCache
    :members:

DatasetCache
=====================
.. autoclass:: qlib.data.cache.DatasetCache
    :members:


Qlib has currently provided `ServerExpressionCache` class and `ServerDatasetCache` class as the cache mechanisms used for QlibServer. The class interface and file structure designed for server cache mechanism is listed below.

ServerExpressionCache
=====================
.. autoclass:: qlib.data.cache.ServerExpressionCache


ServerDatasetCache
====================
.. autoclass:: qlib.data.cache.ServerDatasetCache


Data and cache file structure on server
========================================
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
        [cached data] updated by server when raw data is updated
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
