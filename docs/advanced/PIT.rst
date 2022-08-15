.. _pit:

============================
(P)oint-(I)n-(T)ime Database
============================
.. currentmodule:: qlib


Introduction
------------
Point-in-time data is a very important consideration when performing any sort of historical market analysis.

For example, let’s say we are backtesting a trading strategy and we are using the past five years of historical data as our input.
Our model is assumed to trade once a day, at the market close, and we’ll say we are calculating the trading signal for 1 January 2020 in our backtest. At that point, we should only have data for 1 January 2020, 31 December 2019, 30 December 2019 etc.

In financial data (especially financial reports), the same piece of data may be amended for multiple times overtime.  If we only use the latest version for historical backtesting, data leakage will happen.
Point-in-time database is designed for solving this problem to make sure user get the right version of data at any historical timestamp. It will keep the performance of online trading and historical backtesting the same.



Data Preparation
----------------

Qlib provides a crawler to help users to download financial data and then a converter to dump the data in Qlib format.
Please follow `scripts/data_collector/pit/README.md <https://github.com/microsoft/qlib/tree/main/scripts/data_collector/pit/>`_ to download and convert data.
Besides, you can find some additional usage examples there.


File-based design for PIT data
------------------------------

Qlib provides a file-based storage for PIT data.

For each feature, it contains 4 columns, i.e. date, period, value, _next.
Each row corresponds to a statement.

The meaning of each feature with filename like `XXX_a.data`:

- `date`: the statement's date of publication.
- `period`: the period of the statement. (e.g. it will be quarterly frequency in most of the markets)
    - If it is an annual period, it will be an integer corresponding to the year
    - If it is an quarterly  periods, it will be an integer like `<year><index of quarter>`.  The last two decimal digits represents the index of quarter. Others represent the year.
- `value`: the described value
- `_next`: the byte index of the next occurance of the field.

Besides the feature data, an index `XXX_a.index` is included to speed up the querying performance

The statements are soted by the `date` in ascending order from the beginning of the file.

.. code-block:: python

    # the data format from XXXX.data
    array([(20070428, 200701, 0.090219  , 4294967295),
           (20070817, 200702, 0.13933   , 4294967295),
           (20071023, 200703, 0.24586301, 4294967295),
           (20080301, 200704, 0.3479    ,         80),
           (20080313, 200704, 0.395989  , 4294967295),
           (20080422, 200801, 0.100724  , 4294967295),
           (20080828, 200802, 0.24996801, 4294967295),
           (20081027, 200803, 0.33412001, 4294967295),
           (20090325, 200804, 0.39011699, 4294967295),
           (20090421, 200901, 0.102675  , 4294967295),
           (20090807, 200902, 0.230712  , 4294967295),
           (20091024, 200903, 0.30072999, 4294967295),
           (20100402, 200904, 0.33546099, 4294967295),
           (20100426, 201001, 0.083825  , 4294967295),
           (20100812, 201002, 0.200545  , 4294967295),
           (20101029, 201003, 0.260986  , 4294967295),
           (20110321, 201004, 0.30739301, 4294967295),
           (20110423, 201101, 0.097411  , 4294967295),
           (20110831, 201102, 0.24825101, 4294967295),
           (20111018, 201103, 0.318919  , 4294967295),
           (20120323, 201104, 0.4039    ,        420),
           (20120411, 201104, 0.403925  , 4294967295),
           (20120426, 201201, 0.112148  , 4294967295),
           (20120810, 201202, 0.26484701, 4294967295),
           (20121026, 201203, 0.370487  , 4294967295),
           (20130329, 201204, 0.45004699, 4294967295),
           (20130418, 201301, 0.099958  , 4294967295),
           (20130831, 201302, 0.21044201, 4294967295),
           (20131016, 201303, 0.30454299, 4294967295),
           (20140325, 201304, 0.394328  , 4294967295),
           (20140425, 201401, 0.083217  , 4294967295),
           (20140829, 201402, 0.16450299, 4294967295),
           (20141030, 201403, 0.23408499, 4294967295),
           (20150421, 201404, 0.319612  , 4294967295),
           (20150421, 201501, 0.078494  , 4294967295),
           (20150828, 201502, 0.137504  , 4294967295),
           (20151023, 201503, 0.201709  , 4294967295),
           (20160324, 201504, 0.26420501, 4294967295),
           (20160421, 201601, 0.073664  , 4294967295),
           (20160827, 201602, 0.136576  , 4294967295),
           (20161029, 201603, 0.188062  , 4294967295),
           (20170415, 201604, 0.244385  , 4294967295),
           (20170425, 201701, 0.080614  , 4294967295),
           (20170728, 201702, 0.15151   , 4294967295),
           (20171026, 201703, 0.25416601, 4294967295),
           (20180328, 201704, 0.32954201, 4294967295),
           (20180428, 201801, 0.088887  , 4294967295),
           (20180802, 201802, 0.170563  , 4294967295),
           (20181029, 201803, 0.25522   , 4294967295),
           (20190329, 201804, 0.34464401, 4294967295),
           (20190425, 201901, 0.094737  , 4294967295),
           (20190713, 201902, 0.        ,       1040),
           (20190718, 201902, 0.175322  , 4294967295),
           (20191016, 201903, 0.25581899, 4294967295)],
          dtype=[('date', '<u4'), ('period', '<u4'), ('value', '<f8'), ('_next', '<u4')])
    # - each row contains 20 byte


    # The data format from XXXX.index.  It consists of two parts
    # 1) the start index of the data. So the first part of the info will be like
    2007
    # 2) the remain index data will be like information below
    #    - The data indicate the **byte index** of first data update of a period.
    #    - e.g. Because the info at both byte 80 and 100 corresponds to 200704. The byte index of first occurance (i.e. 100) is recorded in the data.
    array([         0,         20,         40,         60,        100,
                  120,        140,        160,        180,        200,
                  220,        240,        260,        280,        300,
                  320,        340,        360,        380,        400,
                  440,        460,        480,        500,        520,
                  540,        560,        580,        600,        620,
                  640,        660,        680,        700,        720,
                  740,        760,        780,        800,        820,
                  840,        860,        880,        900,        920,
                  940,        960,        980,       1000,       1020,
                 1060, 4294967295], dtype=uint32)




Known limitations:

- Currently, the PIT database is designed for quarterly or annually factors, which can handle fundamental data of financial reports in most markets.
- Qlib leverage the file name to identify the type of the data. File with name like `XXX_q.data` corresponds to quarterly data. File with name like `XXX_a.data` corresponds to annual data.
- The caclulation of PIT is not performed in the optimal way. There is great potential to boost the performance of PIT data calcuation.
