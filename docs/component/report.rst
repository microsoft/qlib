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
    - Axis Y: Accumulated value
    - The shaded part above: Maximum drawdown corresponding to `cum return`
    - The shaded part below: Maximum drawdown corresponding to `cum ex return wo cost` % 

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
    - Axis Y: `Ref($close, -1)/$close - 1` and `score` IC% 

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

    - Cumulative return graphics.
        - Axis X: Trading day
        - Axis Y:
            - Above axis Y: `(((Ref($close, -1)/$close - 1) * weight).sum() / weight.sum()).cumsum()`
            - Below axis Y: Daily weight sum
        - In the **sell** graph, `y < 0` stands for profit; in other cases, `y > 0` stands for profit.
        - In the **buy_minus_sell** graph, the **y** value of the **weight** graph at the bottom is `buy_weight + sell_weight`.
        - In each graph, the **red line** in the histogram on the right represents the average.%                                                                                                                  

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


.. note:: 

    - annual/mdd/sharpe/std graphics
        - Axis X: Trading days are grouped by month
        - Axis Y: monthly(trading date) value

Graphical Result
~~~~~~~~~~~~~~~~~

.. image:: ../_static/img/analysis/risk_analysis_bar.png 

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
        - Axis Y:  Percentage of `'Ref($close, -1)/$close - 1'.rank(ascending=False) / (number of lines on the day) * 100` every trading day. (`ascending=False`: The higher the value, the higher the ranking)%   

.. image:: ../_static/img/analysis/rank_label_hold.png 

.. image:: ../_static/img/analysis/rank_label_buy.png 

.. image:: ../_static/img/analysis/rank_label_sell.png 



Usage of `analysis_model.analysis_model_performance`
-----------------------------------------------------

API
~~~~~

.. automodule:: qlib.contrib.report.analysis_model.analysis_model_performance
    :members:


Graphical Result
~~~~~~~~~~~~~~~~~

.. image:: ../_static/img/analysis/analysis_model_cumulative_return.png 

.. image:: ../_static/img/analysis/analysis_model_long_short.png 

.. image:: ../_static/img/analysis/analysis_model_IC.png 

.. image:: ../_static/img/analysis/analysis_model_monthly_IC.png 

.. image:: ../_static/img/analysis/analysis_model_NDQ.png 

.. image:: ../_static/img/analysis/analysis_model_auto_correlation.png 