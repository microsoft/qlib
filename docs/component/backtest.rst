.. _backtest:

============================================
Intraday Trading: Model&Strategy Testing
============================================
.. currentmodule:: qlib

Introduction
===================

``Intraday Trading`` is designed to test models and strategies, which help users to check the performance of a custom model/strategy.


.. note::

    ``Intraday Trading`` uses ``Order Executor`` to trade and execute orders output by ``Portfolio Strategy``. ``Order Executor`` is a component in `Qlib Framework <../introduction/introduction.html#framework>`_, which can execute orders. ``VWAP Executor`` and ``Close Executor`` is supported by ``Qlib`` now. In the future, ``Qlib`` will support ``HighFreq Executor`` also. 



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

To know more about backtesting with a specific ``Strategy``, please refer to `Portfolio Strategy <strategy.html>`_.

To know more about the prediction score `pred_score` output by ``Forecast Model``, please refer to `Forecast Model: Model Training & Prediction <model.html>`_.

Prediction Score
-----------------

The `prediction score` is a pandas DataFrame. Its index is <datetime(pd.Timestamp), instrument(str)> and it must
contains a `score` column.

A prediction sample is shown as follows.

.. code-block:: python

      datetime instrument     score
    2019-01-04   SH600000 -0.505488
    2019-01-04   SZ002531 -0.320391
    2019-01-04   SZ000999  0.583808
    2019-01-04   SZ300569  0.819628
    2019-01-04   SZ001696 -0.137140
                 ...            ...
    2019-04-30   SZ000996 -1.027618
    2019-04-30   SH603127  0.225677
    2019-04-30   SH603126  0.462443
    2019-04-30   SH603133 -0.302460
    2019-04-30   SZ300760 -0.126383

``Forecast Model`` module can make predictions, please refer to `Forecast Model: Model Training & Prediction <model.html>`_.

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
