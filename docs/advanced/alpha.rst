.. _alpha:
===========================
Building Formulaic Alphas 
===========================
.. currentmodule:: qlib

Introduction
===================

In quantitative trading practice, designing novel factors that can explain and predict future asset returns are of vital importance to the profitability of a strategy. Such factors are usually called alpha factors, or alphas in short.


A formulaic alpha, as the name suggests, is a kind of alpha that can be presented as a formula or a mathematical expression.


Building Formulaic Alphas in ``Qlib``
======================================

In ``Qlib``, users can easily build formulaic alphas.

Example
-----------------

`MACD`, short for moving average convergence/divergence, is a formulaic alpha used in technical analysis of stock prices. It is designed to reveal changes in the strength, direction, momentum, and duration of a trend in a stock's price.

`MACD` can be presented as the following formula:

.. math:: 

    MACD = 2\times (DIF-DEA)

.. note::

    `DIF` means Differential value, which is 12-period EMA minus 26-period EMA.
    
    .. math::

        DIF = \frac{EMA(CLOSE, 12) - EMA(CLOSE, 26)}{CLOSE} 

    `DEA`means a 9-period EMA of the DIF.

    .. math::

        DEA = \frac{EMA(DIF, 9)}{CLOSE}

Users can use ``Data Handler`` to build formulaic alphas `MACD` in qlib:

.. note:: Users need to initialize ``Qlib`` with `qlib.init` first.  Please refer to `initialization <../start/initialization.html>`_.

.. code-block:: python

    >>> from qlib.contrib.estimator.handler import QLibDataHandler
    >>> fields = ['(EMA($close, 12) - EMA($close, 26))/$close - EMA((EMA($close, 12) - EMA($close, 26))/$close, 9)/$close'] # MACD
    >>> names = ['MACD']
    >>> labels = ['Ref($vwap, -2)/Ref($vwap, -1) - 1'] # label
    >>> label_names = ['LABEL']
    >>> data_handler = QLibDataHandler(start_date='2010-01-01', end_date='2017-12-31', fields=fields, names=names, labels=labels, label_names=label_names)
    >>> TRAINER_CONFIG = {
    ...     "train_start_date": "2007-01-01",
    ...     "train_end_date": "2014-12-31",
    ...     "validate_start_date": "2015-01-01",
    ...     "validate_end_date": "2016-12-31",
    ...  "test_start_date": "2017-01-01",
    ...  "test_end_date": "2020-08-01",
    ... }
    >>> feature_train, label_train, feature_validate, label_validate, feature_test, label_test = data_handler.get_split_data(**TRAINER_CONFIG)
    >>> print(feature_train, label_train)
                                MACD
    instrument  datetime            
    SH600004    2012-01-04 -0.030853
                2012-01-05 -0.030452
                2012-01-06 -0.028252
                2012-01-09 -0.024507
                2012-01-10 -0.019744
    ...                         ...
    SZ300273    2014-12-25  0.031339
                2014-12-26  0.029695
                2014-12-29  0.025577
                2014-12-30  0.020493
                2014-12-31  0.017089

    [605882 rows x 1 columns]
                               label
    instrument  datetime            
    SH600004    2012-01-04  0.003021
                2012-01-05  0.017434
                2012-01-06  0.015490
                2012-01-09  0.002324
                2012-01-10 -0.002542
    ...                         ...
    SZ300273    2014-12-25 -0.032454
                2014-12-26 -0.016638
                2014-12-29  0.008263
                2014-12-30 -0.011985
                2014-12-31  0.047797

    [605882 rows x 1 columns]

Reference
===========

To kown more about ``Data Handler``, please refer to `Data Handler <../component/data.html>`_

To kown more about ``Data Api``, please refer to `Data Api <../component/data.html>`_