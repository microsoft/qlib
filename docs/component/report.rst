.. _report:
==========================================
Aanalysis: Evaluation & Results Analysis
==========================================

Introduction
===================

``Aanalysis`` is designed to show the graphical reports of ``Intraday Trading`` , which helps users to evaluate and analyse investment portfolios visually. There are the following graphics to view:

- analysis_position
    - report_graph
    - score_ic_graph
    - cumulative_return_graph
    - risk_analysis_graph
    - rank_label_graph

- analysis_model
    - model_performance_graph


Graphical Reports
===================

Users can run the following code to get all supported reports.

.. code-block:: python

    >>> import qlib.contrib.report as qcr
    >>> print(qcr.GRAPH_NAME_LISt)
    ['analysis_position.report_graph', 'analysis_position.score_ic_graph', 'analysis_position.cumulative_return_graph', 'analysis_position.risk_analysis_graph', 'analysis_position.rank_label_graph', 'analysis_model.model_performance_graph']

.. note::

    For more details, please refer to the function document: similar to ``help(qcr.analysis_position.report_graph)``



Usage&Example
===================

Usage of `analysis_position.report`
-----------------------------------

API
~~~~~~~~~~~~~~~~

.. automodule:: qlib.contrib.report.analysis_position.report
    :members:

Graphical Result
~~~~~~~~~~~~~~~~

.. note:: 

    - Axis X: Trading day
    - Axis Y: 
        - `cum bench`
            Cumulative returns series of benchmark
        - `cum return wo cost`
            Cumulative returns series of portfolio without cost
        - `cum return w cost`
            Cumulative returns series of portfolio with cost
        - `return wo mdd`
            Maximum drawdown series of cumulative return without cost
        - `return w cost mdd`:
            Maximum drawdown series of cumulative return with cost
        - `cum ex return wo cost`
            The `CAR` (cumulative abnormal return) series of the portfolio compared to the benchmark without cost.
        - `cum ex return w cost`
            The `CAR` (cumulative abnormal return) series of the portfolio compared to the benchmark with cost.
        - `turnover`
            Turnover rate series
        - `cum ex return wo cost mdd`
            Drawdown series of `CAR` (cumulative abnormal return) without cost
        - `cum ex return w cost mdd`
            Drawdown series of `CAR` (cumulative abnormal return) with cost
    - The shaded part above: Maximum drawdown corresponding to `cum return wo cost`
    - The shaded part below: Maximum drawdown corresponding to `cum ex return wo cost`

.. image:: ../_static/img/analysis/report.png 


Usage of `analysis_position.score_ic`
-------------------------------------

API
~~~~~~~~~~~~~~~~

.. automodule:: qlib.contrib.report.analysis_position.score_ic
    :members:


Graphical Result
~~~~~~~~~~~~~~~~~

.. note:: 

    - Axis X: Trading day
    - Axis Y: 
        - `ic`
            The `Pearson correlation coefficient` series between `label` and `prediction score`.
            In the above example, the `label` is formulated as `Ref($close, -1)/$close - 1`. Please refer to `Data API Featrue <data.html>`_ for more details.
                
        - `rank_ic`
            The `Spearman's rank correlation coefficient` series between `label` and `prediction score`.

.. image:: ../_static/img/analysis/score_ic.png 


Usage of `analysis_position.cumulative_return`
----------------------------------------------

API
~~~~~~~~~~~~~~~~

.. automodule:: qlib.contrib.report.analysis_position.cumulative_return
    :members:

Graphical Result
~~~~~~~~~~~~~~~~~

.. note:: 

    - Axis X: Trading day
    - Axis Y:
        - Above axis Y: `(((Ref($close, -1)/$close - 1) * weight).sum() / weight.sum()).cumsum()`
        - Below axis Y: Daily weight sum
    - In the **sell** graph, `y < 0` stands for profit; in other cases, `y > 0` stands for profit.
    - In the **buy_minus_sell** graph, the **y** value of the **weight** graph at the bottom is `buy_weight + sell_weight`.
    - In each graph, the **red line** in the histogram on the right represents the average.                                                                                                        

.. image:: ../_static/img/analysis/cumulative_return_buy.png 

.. image:: ../_static/img/analysis/cumulative_return_sell.png 

.. image:: ../_static/img/analysis/cumulative_return_buy_minus_sell.png 

.. image:: ../_static/img/analysis/cumulative_return_hold.png 


Usage of `analysis_position.risk_analysis`
----------------------------------------------

API
~~~~~~~~~~~~~~~~

.. automodule:: qlib.contrib.report.analysis_position.risk_analysis
    :members:


Graphical Result
~~~~~~~~~~~~~~~~~

.. note:: 

    - general graphics
        - `std`
            - `sub_bench`
                The `Standard Deviation` of `CAR` (cumulative abnormal return) without cost.
            - `sub_cost`
                The `Standard Deviation` of `CAR` (cumulative abnormal return) with cost.
        - `annual`
            - `sub_bench`
                The `Annualized Rate` of `CAR` (cumulative abnormal return) without cost.
            - `sub_cost`
                The `Annualized Rate` of `CAR` (cumulative abnormal return) with cost.
        -  `ir`
            - `sub_bench`
                The `Information Ratio` without cost.
            - `sub_cost`
                The `Information Ratio` with cost.
            To kown more about `Information Ratio`, please refer to `Information Ratio – IR <https://www.investopedia.com/terms/i/informationratio.asp>`_.
        -  `mdd`
            - `sub_bench`
                The `Maximum Drawdown` of `CAR` (cumulative abnormal return) without cost.
            - `sub_cost`
                The `Maximum Drawdown` of `CAR` (cumulative abnormal return) with cost.


.. image:: ../_static/img/analysis/risk_analysis_bar.png 

.. note:: 

    - annual/mdd/ir/std graphics
        - Axis X: Trading days grouped by month
        - Axis Y:
            - annual graphics
                - `sub_bench_annual`
                    The `Annualized Rate` series of monthly `CAR` (cumulative abnormal return) without cost.
                - `sub_cost_annual`
                    The `Annualized Rate` series of monthly `CAR` (cumulative abnormal return) with cost.
            - mdd graphics
                - `sub_bench_mdd`
                    The `Maximum Drawdown` series of monthly `CAR` (cumulative abnormal return) without cost.
                - `sub_cost_mdd`
                    The `Maximum Drawdown` series of monthly `CAR` (cumulative abnormal return) with cost.
            - ir graphics
                - `sub_bench_ir`
                    The `Information Ratio` series of monthly `CAR` (cumulative abnormal return) without cost.
                - `sub_cost_ir`
                    The `Information Ratio` series of monthly `CAR` (cumulative abnormal return) with cost.
            - std graphics
                - `sub_bench_mdd`
                    The `Standard Deviation` series of monthly `CAR` (cumulative abnormal return) without cost.
                - `sub_cost_mdd`
                    The `Standard Deviation` series of monthly `CAR` (cumulative abnormal return) with cost.
                

.. image:: ../_static/img/analysis/risk_analysis_annual.png 

.. image:: ../_static/img/analysis/risk_analysis_mdd.png 

.. image:: ../_static/img/analysis/risk_analysis_sharpe.png 

.. image:: ../_static/img/analysis/risk_analysis_std.png 


Usage of `analysis_position.rank_label`
----------------------------------------------

API
~~~~~

.. automodule:: qlib.contrib.report.analysis_position.rank_label
    :members:


Graphical Result
~~~~~~~~~~~~~~~~~

.. note:: 

    - hold/sell/buy graphics:
        - Axis X: Trading day
        - Axis Y: 
            Average `ranking ratio`of `label` for stocks that is held/sold/bought on the trading day.

            In the above example, the `label` is formulated as `Ref($close, -1)/$close - 1`. The `ranking ratio` can be formulated as follows.
            
            .. math::
                
                \frac{Ascending\ Ranking\ of\ 'Ref($close, -1)/$close - 1'}{Number\ of\ Stocks\ on\ the\ Day} \times 100

.. image:: ../_static/img/analysis/rank_label_hold.png 

.. image:: ../_static/img/analysis/rank_label_buy.png 

.. image:: ../_static/img/analysis/rank_label_sell.png 



Usage of `analysis_model.analysis_model_performance`
-----------------------------------------------------

API
~~~~~

.. automodule:: qlib.contrib.report.analysis_model.analysis_model_performance
    :members:


Graphical Results
~~~~~~~~~~~~~~~~~~

.. note::

    - cumulative return graphics
        - `Group1`:
            The `Cumulative Return` series of stocks group with (`ranking ratio` of label <= 20%)
        - `Group2`:
            The `Cumulative Return` series of stocks group with (20% < `ranking ratio` of label <= 40%)
        - `Group3`:
            The `Cumulative Return` series of stocks group with (40% < `ranking ratio` of label <= 60%)
        - `Group4`:
            The `Cumulative Return` series of stocks group with (60% < `ranking ratio` of label <= 80%)
        - `Group5`:
            The `Cumulative Return` series of stocks group with (80% < `ranking ratio` of label)
        - `long-short`:
            The Difference series between `Cumulative Return` of `Group1` and of `Group5`
        - `long-average`
            The Difference series between `Cumulative Return` of `Group1` and average `Cumulative Return` for all stocks.

.. image:: ../_static/img/analysis/analysis_model_cumulative_return.png 

.. note::
    - long-short/long-average
        The distribution of long-short/long-average returns on each trading day


.. image:: ../_static/img/analysis/analysis_model_long_short.png 

.. TODO: ask xiao yang for detial

.. note::
    - Information Coefficient
        The `Pearson correlation coefficient` series between the latest `label` and the `label` `lag` days ago of stocks in portfolio on each trading day.

.. image:: ../_static/img/analysis/analysis_model_IC.png 

.. note::
    - Monthly IC
        Monthly average of the `Information Coefficient`

.. image:: ../_static/img/analysis/analysis_model_monthly_IC.png 

.. note::
    - IC
        The distribution of the `Information Coefficient` on each trading day.
    - IC Normal Dist. Q-Q
        The `Quantile-Quantile Plot` used for the normal distribution of `Information Coefficient` on each trading day.

.. image:: ../_static/img/analysis/analysis_model_NDQ.png 

.. note::
    - Auto Correlation
         The `Pearson correlation coefficient` series between `label` and `prediction score`of stocks in portfolio.

.. image:: ../_static/img/analysis/analysis_model_auto_correlation.png 