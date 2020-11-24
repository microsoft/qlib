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

    >> from qlib.data.dataset.handler import QLibDataHandler
    >> MACD_EXP = '(EMA($close, 12) - EMA($close, 26))/$close - EMA((EMA($close, 12) - EMA($close, 26))/$close, 9)/$close'
    >> fields = [MACD_EXP] # MACD
    >> names = ['MACD']
    >> labels = ['$close'] # label
    >> label_names = ['LABEL']
    >> data_handler = QLibDataHandler(start_date='2010-01-01', end_date='2017-12-31', fields=fields, names=names, labels=labels, label_names=label_names)
    >> TRAINER_CONFIG = {
    ..     "train_start_date": "2007-01-01",
    ..     "train_end_date": "2014-12-31",
    ..     "validate_start_date": "2015-01-01",
    ..     "validate_end_date": "2016-12-31",
    ..  "test_start_date": "2017-01-01",
    ..  "test_end_date": "2020-08-01",
    .. }
    >> feature_train, label_train, feature_validate, label_validate, feature_test, label_test = data_handler.get_split_data(**TRAINER_CONFIG)
    >> print(feature_train, label_train)
                            MACD
    instrument  datetime            
    SH600000    2010-01-04 -0.008625
                2010-01-05 -0.007234
                2010-01-06 -0.007693
                2010-01-07 -0.009633
                2010-01-08 -0.009891
    ...                         ...
    SZ300251    2014-12-25  0.043072
                2014-12-26  0.041345
                2014-12-29  0.042733
                2014-12-30  0.042066
                2014-12-31  0.036299

    [322025 rows x 1 columns]    
                            LABEL
    instrument  datetime            
    SH600000    2010-01-04  4.260015
                2010-01-05  4.292182
                2010-01-06  4.207747
                2010-01-07  4.113258
                2010-01-08  4.159496
    ...                         ...
    SZ300251    2014-12-25  4.343212
                2014-12-26  4.470587
                2014-12-29  4.762474
                2014-12-30  4.369748
                2014-12-31  4.182222

    [322025 rows x 1 columns]

Reference
===========

To learn more about ``Data Handler``, please refer to `Data Handler <../component/data.html>`_

To learn more about ``Data API``, please refer to `Data API <../component/data.html>`_
