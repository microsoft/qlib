.. _getdata:
=============================
Data Retrieval
=============================

.. currentmodule:: qlib

Introduction
====================

Users can get stock data with ``Qlib``. Following examples demonstrates the basic user interface.

Examples
====================


``QLib`` Initialization:

.. note:: In order to get the data, users need to initialize ``Qlib`` with `qlib.init` first. Please refer to `initialization <initialization.rst>`_.

If user followed steps in `initialization <initialization.rst>`_ and downloaded the data, use should use following code to initialize qlib 

.. code-block:: python

    >>> import qlib
    >>> qlib.init(provider_uri='~/.qlib/qlib_data/cn_data')


Load trading calendar with the given time range and frequency:

.. code-block:: python
		
   >>> from qlib.data import D
   >>> D.calendar(start_time='2010-01-01', end_time='2017-12-31', freq='day')[:2]
   [Timestamp('2010-01-04 00:00:00'), Timestamp('2010-01-05 00:00:00')]

Parse a given market name into a stockpool config:

.. code-block:: python

   >>> from qlib.data import D
   >>> D.instruments(market='all')
   {'market': 'all', 'filter_pipe': []}

Load instruments of certain stockpool in the given time range:

.. code-block:: python
		
   >>> from qlib.data import D
   >>> instruments = D.instruments(market='csi300')
   >>> D.list_instruments(instruments=instruments, start_time='2010-01-01', end_time='2017-12-31', as_list=True)[:6]


Load dynamic instruments from a base market according to a name filter

.. code-block:: python

   >>> from qlib.data import D
   >>> from qlib.data.filter import NameDFilter
   >>> nameDFilter = NameDFilter(name_rule_re='SH[0-9]{4}55')
   >>> instruments = D.instruments(market='csi300', filter_pipe=[nameDFilter])
   >>> D.list_instruments(instruments=instruments, start_time='2015-01-01', end_time='2016-02-15', as_list=True)

Load dynamic instruments from a base market according to an expression filter

.. code-block:: python

   >>> from qlib.data import D
   >>> from qlib.data.filter import ExpressionDFilter
   >>> expressionDFilter = ExpressionDFilter(rule_expression='$close>100')
   >>> instruments = D.instruments(market='csi300', filter_pipe=[expressionDFilter])
   >>> D.list_instruments(instruments=instruments, start_time='2015-01-01', end_time='2016-02-15', as_list=True)

For more details about filter, please refer to API Reference: `filter API <../reference/api.html#filter>`_

Load features of certain instruments in given time range:

.. code-block:: python
		
   >>> from qlib.data import D
   >>> instruments = ['SH600000']
   >>> fields = ['$close', '$volume', 'Ref($close, 1)', 'Mean($close, 3)', '$high-$low']
   >>> D.features(instruments, fields, start_time='2010-01-01', end_time='2017-12-31', freq='day').head()
		                     $close      $volume      Ref($close,1)   Mean($close,3)  \
   instrument  datetime
   SH600000    2010-01-04  81.809998   17144536.0         NaN       81.809998
	            2010-01-05  82.419998   29827816.0   81.809998       82.114998
               2010-01-06  80.800003   25070040.0   82.419998       81.676666
               2010-01-07  78.989998   22077858.0   80.800003       80.736666
               2010-01-08  79.879997   17019168.0   78.989998       79.889999

                           Sub($high,$low)
   instrument  datetime
   SH600000    2010-01-04  2.741158
	            2010-01-05  3.049736
               2010-01-06  1.621399
               2010-01-07  2.856926
               2010-01-08  1.930397
               2010-01-08  1.930397

Load features of certain stockpool in given time range:

.. note:: With cache enabled, the qlib data server will cache data on all the time for the requested stockpool and fields, it may take longer to process the request for the first time than that without cache. But after the first time, requests with the same stockpool and fields will hit the cache and be processed faster even the requested time period changes.

.. code-block:: python

   >>> from qlib.data import D
   >>> from qlib.data.filter import NameDFilter, ExpressionDFilter
   >>> nameDFilter = NameDFilter(name_rule_re='SH[0-9]{4}55')
   >>> expressionDFilter = ExpressionDFilter(rule_expression='($close/$factor)>100')
   >>> instruments = D.instruments(market='csi300', filter_pipe=[nameDFilter, expressionDFilter])
   >>> fields = ['$close', '$volume', 'Ref($close, 1)', 'Mean($close, 3)', '$high-$low']
   >>> D.features(instruments, fields, start_time='2010-01-01', end_time='2017-12-31', freq='day').head()

   		                    $close	    $volume	        Ref($close, 1)	\
   instrument datetime
   SH600655	  2015-06-15	4342.160156	258706.359375	4530.459961
              2015-06-16	4409.270020	257349.718750	4342.160156
              2015-06-17	4312.330078	235214.890625	4409.270020
              2015-06-18	4086.729980	196772.859375	4312.330078
              2015-06-19	3678.250000	182916.453125	4086.729980
                            Mean($close, 3)	 high− low
   instrument datetime
   SH600655   2015-06-15    4480.743327	     285.251465
              2015-06-16    4427.296712	     298.301270
              2015-06-16    4354.586751	     356.098145
              2015-06-16    4269.443359	     363.554932
              2015-06-16    4025.770020	     368.954346


.. note:: When calling D.features() at client, use parameter 'disk_cache=0' to skip dataset cache, use 'disk_cache=1' to generate and use dataset cache. In addition, when calling at server, you can use 'disk_cache=2' to update the dataset cache.

API
====================
To know more about how to use the Data, go to API Reference: `Data API <../reference/api.html#Data>`_
