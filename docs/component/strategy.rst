.. _strategy:

========================================
Portfolio Strategy: Portfolio Management
========================================
.. currentmodule:: qlib

Introduction
===================

``Portfolio Strategy`` is designed to adopt different portfolio strategies, which means that users can adopt different algorithms to generate investment portfolios based on the prediction scores of the ``Forecast Model``. Users can use the ``Portfolio Strategy`` in an automatic workflow by ``Workflow`` module, please refer to `Workflow: Workflow Management <workflow.html>`_.  

Because the components in ``Qlib`` are designed in a loosely-coupled way, ``Portfolio Strategy`` can be used as an independent module also.

``Qlib`` provides several implemented portfolio strategies. Also, ``Qlib`` supports custom strategy, users can customize strategies according to their own needs.

Base Class & Interface
======================

BaseStrategy
------------------

Qlib provides a base class ``qlib.contrib.strategy.BaseStrategy``. All strategy classes need to inherit the base class and implement its interface.

- `get_risk_degree`
    Return the proportion of your total value you will use in investment. Dynamically risk_degree will result in Market timing.

- `generate_order_list`
    Return the order list. 

Users can inherit `BaseStrategy` to customize their strategy class.

WeightStrategyBase
--------------------

Qlib also provides a class ``qlib.contrib.strategy.WeightStrategyBase`` that is a subclass of `BaseStrategy`. 

`WeightStrategyBase` only focuses on the target positions, and automatically generates an order list based on positions. It provides the `generate_target_weight_position` interface.

- `generate_target_weight_position`
    - According to the current position and trading date to generate the target position. The cash is not considered in
      the output weight distribution.
    - Return the target position.

    .. note::
        Here the `target position` means the target percentage of total assets.

`WeightStrategyBase` implements the interface `generate_order_list`, whose processions is as follows.

- Call `generate_target_weight_position` method to generate the target position.
- Generate the target amount of stocks from the target position.
- Generate the order list from the target amount

Users can inherit `WeightStrategyBase` and implement the interface `generate_target_weight_position` to customize their strategy class, which only focuses on the target positions.

Implemented Strategy
====================

Qlib provides a implemented strategy classes named `TopkDropoutStrategy`.

TopkDropoutStrategy
------------------
`TopkDropoutStrategy` is a subclass of `BaseStrategy` and implement the interface `generate_order_list` whose process is as follows.

- Adopt the ``Topk-Drop`` algorithm to calculate the target amount of each stock

    .. note::
        ``Topk-Drop`` algorithm：

        - `Topk`: The number of stocks held
        - `Drop`: The number of stocks sold on each trading day
        
        Currently, the number of held stocks is `Topk`.
        On each trading day, the `Drop` number of held stocks with the worst `prediction score` will be sold, and the same number of unheld stocks with the best `prediction score` will be bought.
        
        .. image:: ../_static/img/topk_drop.png
            :alt: Topk-Drop

        ``TopkDrop`` algorithm sells `Drop` stocks every trading day, which guarantees a fixed turnover rate.
        
- Generate the order list from the target amount

Usage & Example
====================
``Portfolio Strategy`` can be specified in the ``Intraday Trading(Backtest)``, the example is as follows.

- daily

    .. code-block:: python

        from pprint import pprint

        import qlib
        import pandas as pd
        from qlib.utils.time import Freq
        from qlib.utils import flatten_dict
        from qlib.contrib.evaluate import backtest_daily
        from qlib.contrib.evaluate import risk_analysis
        from qlib.contrib.strategy import TopkDropoutStrategy

        # init qlib
        qlib.init(provider_uri=<qlib data dir>)

        CSI300_BENCH = "SH000300"
        STRATEGY_CONFIG = {
            "topk": 50,
            "n_drop": 5,
            # pred_score, pd.Series
            "signal": pred_score,
        }


        strategy_obj = TopkDropoutStrategy(**STRATEGY_CONFIG)
        report_normal, positions_normal = backtest_daily(
            start_time="2017-01-01", end_time="2020-08-01", strategy=strategy_obj
        )
        analysis = dict()
        analysis["excess_return_without_cost"] = risk_analysis(
            report_normal["return"] - report_normal["bench"], freq=analysis_freq
        )
        analysis["excess_return_with_cost"] = risk_analysis(
            report_normal["return"] - report_normal["bench"] - report_normal["cost"], freq=analysis_freq
        )

        analysis_df = pd.concat(analysis)  # type: pd.DataFrame
        pprint(analysis_df)



- nested decision execution

    .. code-block:: python

        from pprint import pprint

        import qlib
        import pandas as pd
        from qlib.utils.time import Freq
        from qlib.utils import flatten_dict
        from qlib.backtest import backtest, executor
        from qlib.contrib.evaluate import risk_analysis
        from qlib.contrib.strategy import TopkDropoutStrategy

        # init qlib
        qlib.init(provider_uri=<qlib data dir>)

        CSI300_BENCH = "SH000300"
        FREQ = "day"
        STRATEGY_CONFIG = {
            "topk": 50,
            "n_drop": 5,
            # pred_score, pd.Series
            "signal": pred_score,
        }

        EXECUTOR_CONFIG = {
            "time_per_step": "day",
            "generate_portfolio_metrics": True,
        }

        backtest_config = {
            "start_time": "2017-01-01",
            "end_time": "2020-08-01",
            "account": 100000000,
            "benchmark": CSI300_BENCH,
            "exchange_kwargs": {
                "freq": FREQ,
                "limit_threshold": 0.095,
                "deal_price": "close",
                "open_cost": 0.0005,
                "close_cost": 0.0015,
                "min_cost": 5,
            },
        }

        # strategy object
        strategy_obj = TopkDropoutStrategy(**STRATEGY_CONFIG)
        # executor object
        executor_obj = executor.SimulatorExecutor(**EXECUTOR_CONFIG)
        # backtest
        portfolio_metric_dict, indicator_dict = backtest(executor=executor_obj, strategy=strategy_obj, **backtest_config)
        analysis_freq = "{0}{1}".format(*Freq.parse(FREQ))
        # backtest info
        report_normal, positions_normal = portfolio_metric_dict.get(analysis_freq)

        # analysis
        analysis = dict()
        analysis["excess_return_without_cost"] = risk_analysis(
            report_normal["return"] - report_normal["bench"], freq=analysis_freq
        )
        analysis["excess_return_with_cost"] = risk_analysis(
            report_normal["return"] - report_normal["bench"] - report_normal["cost"], freq=analysis_freq
        )

        analysis_df = pd.concat(analysis)  # type: pd.DataFrame
        # log metrics
        analysis_dict = flatten_dict(analysis_df["risk"].unstack().T.to_dict())
        # print out results
        pprint(f"The following are analysis results of benchmark return({analysis_freq}).")
        pprint(risk_analysis(report_normal["bench"], freq=analysis_freq))
        pprint(f"The following are analysis results of the excess return without cost({analysis_freq}).")
        pprint(analysis["excess_return_without_cost"])
        pprint(f"The following are analysis results of the excess return with cost({analysis_freq}).")
        pprint(analysis["excess_return_with_cost"])


To know more about the `prediction score` `pred_score` output by ``Forecast Model``, please refer to `Forecast Model: Model Training & Prediction <model.html>`_.

To know more about ``Intraday Trading``, please refer to `Intraday Trading: Model&Strategy Testing <backtest.html>`_.

Reference
===================
To know more about ``Portfolio Strategy``, please refer to `Strategy API <../reference/api.html#module-qlib.contrib.strategy.strategy>`_.
