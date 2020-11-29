.. _backtest:

============================================
Intraday Trading: Model&Strategy Testing
============================================
.. currentmodule:: qlib

Introduction
===================

``Intraday Trading`` is designed to test models and strategies, which help users to check the performance of a custom model/strategy.


.. note::

    ``Intraday Trading`` uses ``Order Executor`` to trade and execute orders output by ``Interday Strategy``. ``Order Executor`` is a component in `Qlib Framework <../introduction/introduction.html#framework>`_, which can execute orders. ``Vwap Executor`` and ``Close Executor`` is supported by ``Qlib`` now. In the future, ``Qlib`` will support ``HighFreq Executor`` also. 



Example
===========================

Users need to generate a `prediction score`(a pandas DataFrame) with MultiIndex<instrument, datetime> and a `score` column. And users need to assign a strategy used in backtest, if strategy is not assigned,
a `TopkDropoutStrategy` strategy with `(topk=50, n_drop=5, risk_degree=0.95, limit_threshold=0.0095)` will be used.
If ``Strategy`` module is not users' interested part, `TopkDropoutStrategy` is enough. 

The simple example of the default strategy is as follows.

.. code-block:: python

    from qlib.contrib.evaluate import backtest
    # pred_score is the prediction score
    report, positions = backtest(pred_score, topk=50, n_drop=0.5, verbose=False, limit_threshold=0.0095)

To know more about backtesting with a specific ``Strategy``, please refer to `Strategy <strategy.html>`_.

To know more about the prediction score `pred_score` output by ``Interday Model``, please refer to `Interday Model: Model Training & Prediction <model.html>`_.

Prediction Score
-----------------

The `prediction score` is a pandas DataFrame. Its index is <instrument(str), datetime(pd.Timestamp)> and it must
contains a `score` column.

A prediction sample is shown as follows.

.. code-block:: python

    instrument datetime   score
    SH600000   2019-01-04 -0.505488
    SZ002531   2019-01-04 -0.320391
    SZ000999   2019-01-04  0.583808
    SZ300569   2019-01-04  0.819628
    SZ001696   2019-01-04 -0.137140
    ...                         ...
    SZ000996   2019-04-30 -1.027618
    SH603127   2019-04-30  0.225677
    SH603126   2019-04-30  0.462443
    SH603133   2019-04-30 -0.302460
    SZ300760   2019-04-30 -0.126383

``Interday Model`` module can make predictions, please refer to `Interday Model: Model Training & Prediction <model.html>`_.

Backtest Result
------------------

The backtest results are in the following form:

.. code-block:: python

                                                      risk
    excess_return_without_cost mean               0.000605
                               std                0.005481
                               annualized_return  0.152373
                               information_ratio  1.751319
                               max_drawdown      -0.059055
    excess_return_with_cost    mean               0.000410
                               std                0.005478
                               annualized_return  0.103265
                               information_ratio  1.187411
                               max_drawdown      -0.075024



- `excess_return_without_cost`
    - `mean`
        Mean value of the `CAR` (cumulative abnormal return) without cost
    - `std`
        The `Standard Deviation` of `CAR` (cumulative abnormal return) without cost.
    - `annualized_return`
        The `Annualized Rate` of `CAR` (cumulative abnormal return) without cost.
    - `information_ratio`
        The `Information Ratio` without cost. please refer to `Information Ratio – IR <https://www.investopedia.com/terms/i/informationratio.asp>`_.
    - `max_drawdown`
        The `Maximum Drawdown` of `CAR` (cumulative abnormal return) without cost, please refer to `Maximum Drawdown (MDD) <https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp>`_.

- `excess_return_with_cost`
    - `mean`
        Mean value of the `CAR` (cumulative abnormal return) series with cost
    - `std`
        The `Standard Deviation` of `CAR` (cumulative abnormal return) series with cost.
    - `annualized_return`
        The `Annualized Rate` of `CAR` (cumulative abnormal return) with cost.
    - `information_ratio`
        The `Information Ratio` with cost. please refer to `Information Ratio – IR <https://www.investopedia.com/terms/i/informationratio.asp>`_.
    - `max_drawdown`
        The `Maximum Drawdown` of `CAR` (cumulative abnormal return) with cost, please refer to `Maximum Drawdown (MDD) <https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp>`_.



Reference
==============

To know more about ``Intraday Trading``, please refer to `Intraday Trading <../reference/api.html#module-qlib.contrib.evaluate>`_.
