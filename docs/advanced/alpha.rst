.. _alpha:

=========================
Building Formulaic Alphas
=========================
.. currentmodule:: qlib

Introduction
============

In quantitative trading practice, designing novel factors that can explain and predict future asset returns are of vital importance to the profitability of a strategy. Such factors are usually called alpha factors, or alphas in short.


A formulaic alpha, as the name suggests, is a kind of alpha that can be presented as a formula or a mathematical expression.


Building Formulaic Alphas in ``Qlib``
=====================================

In ``Qlib``, users can easily build formulaic alphas.

Example
-------

`MACD`, short for moving average convergence/divergence, is a formulaic alpha used in technical analysis of stock prices. It is designed to reveal changes in the strength, direction, momentum, and duration of a trend in a stock's price.

`MACD` can be presented as the following formula:

.. math::

    MACD = 2\times (DIF-DEA)

.. note::

    `DIF` means Differential value, which is 12-period EMA minus 26-period EMA.

    .. math::

        DIF = \frac{EMA(CLOSE, 12) - EMA(CLOSE, 26)}{CLOSE}

    `DEA` means a 9-period EMA of the DIF.

    .. math::

        DEA = EMA(DIF, 9)

Users can use ``Data Handler`` to build formulaic alphas `MACD` in qlib:

.. note:: Users need to initialize ``Qlib`` with `qlib.init` first.  Please refer to `initialization <../start/initialization.html>`_.

.. code-block:: python

    >> from qlib.data.dataset.loader import QlibDataLoader
    >> MACD_EXP = '2 * ((EMA($close, 12) - EMA($close, 26))/$close - EMA((EMA($close, 12) - EMA($close, 26))/$close, 9))'
    >> fields = [MACD_EXP] # MACD
    >> names = ['MACD']
    >> labels = ['Ref($close, -2)/Ref($close, -1) - 1'] # label
    >> label_names = ['LABEL']
    >> data_loader_config = {
    ..     "feature": (fields, names),
    ..     "label": (labels, label_names)
    .. }
    >> data_loader = QlibDataLoader(config=data_loader_config)
    >> df = data_loader.load(instruments='csi300', start_time='2010-01-01', end_time='2017-12-31')
    >> print(df)
                            feature     label
                               MACD     LABEL
    datetime   instrument
    2010-01-04 SH600000    0.008781 -0.019672
               SH600004    0.006699 -0.014721
               SH600006    0.005714  0.002911
               SH600008    0.000798  0.009818
               SH600009    0.017015 -0.017758
    ...                         ...       ...
    2017-12-29 SZ300124    0.015071 -0.005074
               SZ300136   -0.015466  0.056352
               SZ300144    0.013082  0.011853
               SZ300251   -0.001026  0.021739
               SZ300315   -0.007559  0.012455

.. _expression_syntax:

Expression syntax and migration
===============================

Feature expressions are a restricted language, not general Python.
``Qlib`` interprets their syntax and calls registered operators rather than evaluating arbitrary Python code.
Standard Alpha158/Alpha360 feature definitions and registered custom operators remain supported.

Supported expressions include:

.. code-block:: text

    $close
    Ref($close, 1) / $close - 1
    Mean($close, 2 + 3)
    $close / (1 + 0.01)
    If(Gt($close, $open), $close, $open)
    ($close > $open) & ($volume > 0)

You can use feature references (``$field`` and point-in-time ``$$field``), registered operator calls, arithmetic on expressions, and single comparisons.
Operator arguments can include literals, lists, tuples, and named arguments where the operator accepts them.
Numeric parameter arithmetic is supported, including ``+``, ``-``, ``*``, ``/``, ``//``, ``%``, and ``**``.
Constant arithmetic requires real numbers; integers are limited to 4096 bits and the absolute value of a constant exponent is limited to 4096.
String/list expansion and complex-valued constant arithmetic are rejected.
The final result must be a Qlib ``Expression`` object, not a standalone constant.

When migrating expressions that relied on Python syntax:

* Replace attribute access or indexing with supported operators, or move the logic into a trusted custom operator.
* Use ``&`` and ``|`` with parenthesized comparisons instead of Python ``and`` and ``or``.
  Write ``($close > 0) & ($close < 10)`` instead of the chained comparison ``0 < $close < 10``.
* Use explicit operator arguments instead of ``*args`` or ``**kwargs`` expansion.
  Lambdas, comprehensions, arbitrary function calls, and constant-only comparisons are not supported.

Unsupported syntax and unknown operator names raise ``qlib.data.expression_parser.ExpressionSyntaxError``, a subclass of ``ValueError``.
Operator-specific argument validation still applies.

Register custom operators before use, for example through ``qlib.init(custom_ops=[...])``; merely making a Python function importable does not make it an expression operator.
See ``tests/test_register_ops.py`` for an example.
For file-based custom operators, also configure :ref:`trusted_module_roots`.

.. note::

    Custom operator code must be trusted. The expression language restricts syntax, but does not sandbox registered operators or impose general resource limits on expression evaluation.

Reference
=========

To learn more about ``Data Loader``, please refer to `Data Loader <../component/data.html#data-loader>`_

To learn more about ``Data API``, please refer to `Data API <../component/data.html>`_
