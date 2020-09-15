.. _strategy:
==========================
Strategy: Portfolio Management
==========================
.. currentmodule:: qlib

Introduction
===================

By ``Strategy``, users can adopt different trading strategies, which means that users can use different algorithms to generate investment portfolios based on the predicted scores of the ``Model`` module.

``Qlib`` provides several trading strategy classes, users can customize strategies according to their own needs also.

Base Class & Interface
=====================

BaseStrategy
------------------

Qlib provides a base class ``qlib.contrib.strategy.BaseStrategy``. All strategy classes need to inherit the base class and implement its interface.

- `get_risk_degree`
    Return the proportion of your total value you will use in investment. Dynamically risk_degree will result in Market timing.

- `generate_order_list`
    Rerturn the order list. 

User can inherit 'BaseStrategy' to costomize their strategy class.

WeightStrategyBase
--------------------

Qlib alse provides a class ``qlib.contrib.strategy.WeightStrategyBase`` that is a subclass of `BaseStrategy`. 

`WeightStrategyBase` only focuses on the target positions, and automatically generates an order list based on positions. It provides the `generate_target_weight_position` interface.

- `generate_target_weight_position`
    According to the current position and trading date to generate the target position.
     
    .. note:: The cash is not considered.
    Return the target position.

`WeightStrategyBase` implements the interface `generate_order_list`, whose process is as follows.

- Call `generate_target_weight_position` method to generate the target position.
- Generate the target amount of stocks from the target position.
- Generate the order list from the target amount

User can inherit `WeightStrategyBase` and implement the inteface `generate_target_weight_position` to costomize their strategy class, which focuses on the target positions.

Implemented Strategy
====================

Qlib provides several implemented strategy classes, such as `TopkWeightStrategy`, `TopkAmountStrategy` and `TopkDropoutStrategy`.

TopkWeightStrategy
------------------
`TopkWeightStrategy` is a subclass of `WeightStrategyBase` and implements the interface `generate_target_weight_position`.

The implemented interface `generate_target_weight_position` adopts the ``Topk`` algorithm to calculate the target position, it ensures that the weight of each stock is as even as possible.

.. note:: 
    ``TopK`` algorithm: Define a threshold `margin`. On each trading day, the stocks with the predicted scores behind `margin` will be sold, and then the stocks with the best predicted scores will be bought to maintain the number of stocks at k.



TopkAmountStrategy
------------------
`TopkAmountStrategy` is a subclass of `BaseStrategy` and implement the interface `generate_order_list` whose process is as follows.

- Adopt the the ``Topk`` algorithm to calculate the target amount of each stock
- Generate the order list from the target amount



TopkDropoutStrategy
------------------
`TopkDropoutStrategy` is a subclass of `BaseStrategy` and implement the interface `generate_order_list` whose process is as follows.

- Adopt the the ``TopkDropout`` algorithm to calculate the target amount of each stock

    .. note::

        ``TopkDropout`` algorithm: On each trading day, the held stocks with the worst predicted scores will be sold, and then stocks with the best predicted scores will be bought to maintain the number of stocks at k. Because a fixed number of stocks are sold and bought every day, this algorithm can make the turnover rate a fixed value.

- Generate the order list from the target amount

Example
====================
``Strategy`` can be specified in the ``Backtest`` module, the example is as follows.

.. code-block:: python

    from qlib.contrib.strategy.strategy import TopkAmountStrategy
    from qlib.contrib.evaluate import backtest
    STRATEGY_CONFIG = {
        "topk": 50,
        "buffer_margin": 230,
    }
    BACKTEST_CONFIG = {
        "verbose": False,
        "limit_threshold": 0.095,
        "account": 100000000,
        "benchmark": BENCHMARK,
        "deal_price": "vwap",
    }

    # use default strategy
    # custom Strategy, refer to: TODO: Strategy api url
    strategy = TopkAmountStrategy(**STRATEGY_CONFIG)
    report_normal, positions_normal = backtest(
        pred_score, strategy=strategy, **BACKTEST_CONFIG
    )

Also, the above example has been given in ``examples.estimator.train_backtest_analyze.ipynb``.

To know more about ``Backtest``, please refer to `Backtest: Model&Strategy Testing <backtest.html>`_.

Api
===================
Please refer to `Strategy Api <../reference/api.html>`_.
