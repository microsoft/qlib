.. _getdata:

==============
Data Retrieval
==============

.. currentmodule:: qlib

Introduction
============

Users can get stock data with ``Qlib``. The following examples demonstrate the basic user interface.

Examples
========


``QLib`` Initialization:

.. note:: In order to get the data, users need to initialize ``Qlib`` with `qlib.init` first. Please refer to `initialization <initialization.html>`_.

If users followed steps in `initialization <initialization.html>`_ and downloaded the data, they should use the following code to initialize qlib

.. code-block:: python

    >> import qlib
    >> qlib.init(provider_uri='~/.qlib/qlib_data/cn_data')


Load trading calendar with given time range and frequency:

.. code-block:: python

   >> from qlib.data import D
   >> D.calendar(start_time='2010-01-01', end_time='2017-12-31', freq='day')[:2]
   [Timestamp('2010-01-04 00:00:00'), Timestamp('2010-01-05 00:00:00')]

Parse a given market name into a stock pool config:

.. code-block:: python

   >> from qlib.data import D
   >> D.instruments(market='all')
   {'market': 'all', 'filter_pipe': []}

Load instruments of certain stock pool in the given time range:

.. code-block:: python

   >> from qlib.data import D
   >> instruments = D.instruments(market='csi300')
   >> D.list_instruments(instruments=instruments, start_time='2010-01-01', end_time='2017-12-31', as_list=True)[:6]
   ['SH600036', 'SH600110', 'SH600087', 'SH600900', 'SH600089', 'SZ000912']

Load dynamic instruments from a base market according to a name filter

.. code-block:: python

   >> from qlib.data import D
   >> from qlib.data.filter import NameDFilter
   >> nameDFilter = NameDFilter(name_rule_re='SH[0-9]{4}55')
   >> instruments = D.instruments(market='csi300', filter_pipe=[nameDFilter])
   >> D.list_instruments(instruments=instruments, start_time='2015-01-01', end_time='2016-02-15', as_list=True)
   ['SH600655', 'SH601555']

Load dynamic instruments from a base market according to an expression filter

.. code-block:: python

   >> from qlib.data import D
   >> from qlib.data.filter import ExpressionDFilter
   >> expressionDFilter = ExpressionDFilter(rule_expression='$close>2000')
   >> instruments = D.instruments(market='csi300', filter_pipe=[expressionDFilter])
   >> D.list_instruments(instruments=instruments, start_time='2015-01-01', end_time='2016-02-15', as_list=True)
   ['SZ000651', 'SZ000002', 'SH600655', 'SH600570']

For more details about filter, please refer `Filter API <../component/data.html>`_.

Load features of certain instruments in a given time range:

.. code-block:: python

   >> from qlib.data import D
   >> instruments = ['SH600000']
   >> fields = ['$close', '$volume', 'Ref($close, 1)', 'Mean($close, 3)', '$high-$low']
   >> D.features(instruments, fields, start_time='2010-01-01', end_time='2017-12-31', freq='day').head().to_string()
   '                           $close     $volume  Ref($close, 1)  Mean($close, 3)  $high-$low
   ... instrument  datetime
   ... SH600000    2010-01-04  86.778313  16162960.0       88.825928        88.061483    2.907631
   ...             2010-01-05  87.433578  28117442.0       86.778313        87.679273    3.235252
   ...             2010-01-06  85.713585  23632884.0       87.433578        86.641825    1.720009
   ...             2010-01-07  83.788803  20813402.0       85.713585        85.645322    3.030487
   ...             2010-01-08  84.730675  16044853.0       83.788803        84.744354    2.047623'

Load features of certain stock pool in a given time range:

.. note:: With cache enabled, the qlib data server will cache data all the time for the requested stock pool and fields, it may take longer to process the request for the first time than that without cache. But after the first time, requests with the same stock pool and fields will hit the cache and be processed faster even the requested time period changes.

.. code-block:: python

   >> from qlib.data import D
   >> from qlib.data.filter import NameDFilter, ExpressionDFilter
   >> nameDFilter = NameDFilter(name_rule_re='SH[0-9]{4}55')
   >> expressionDFilter = ExpressionDFilter(rule_expression='$close>Ref($close,1)')
   >> instruments = D.instruments(market='csi300', filter_pipe=[nameDFilter, expressionDFilter])
   >> fields = ['$close', '$volume', 'Ref($close, 1)', 'Mean($close, 3)', '$high-$low']
   >> D.features(instruments, fields, start_time='2010-01-01', end_time='2017-12-31', freq='day').head().to_string()
   '                              $close        $volume  Ref($close, 1)  Mean($close, 3)  $high-$low
   ... instrument  datetime
   ... SH600655    2010-01-04  2699.567383  158193.328125     2619.070312      2626.097738  124.580566
   ...             2010-01-08  2612.359619   77501.406250     2584.567627      2623.220133   83.373047
   ...             2010-01-11  2712.982422  160852.390625     2612.359619      2636.636556  146.621582
   ...             2010-01-12  2788.688232  164587.937500     2712.982422      2704.676758  128.413818
   ...             2010-01-13  2790.604004  145460.453125     2788.688232      2764.091553  128.413818'


For more details about features, please refer `Feature API <../component/data.html>`_.

.. note:: When calling `D.features()` at the client, use parameter `disk_cache=0` to skip dataset cache, use `disk_cache=1` to generate and use dataset cache. In addition, when calling at the server, users can use `disk_cache=2` to update the dataset cache.


When you are building complicated expressions, implementing all the expressions in a single string may not be easy.
For example, it looks quite long and complicated:

.. code-block:: python

   >> from qlib.data import D
   >> data = D.features(["sh600519"], ["(($high / $close) + ($open / $close)) * (($high / $close) + ($open / $close)) / (($high / $close) + ($open / $close))"], start_time="20200101")


But using string is not the only way to implement the expression. You can also implement expression by code.
Here is an exmaple which does the same thing as above examples.


.. code-block:: python

   >> from qlib.data.ops import *
   >> f1 = Feature("high") / Feature("close")
   >> f2 = Feature("open") / Feature("close")
   >> f3 = f1 + f2
   >> f4 = f3 * f3 / f3

   >> data = D.features(["sh600519"], [f4], start_time="20200101")
   >> data.head()


API
===
To know more about how to use the Data, go to API Reference: `Data API <../reference/api.html#data>`_
