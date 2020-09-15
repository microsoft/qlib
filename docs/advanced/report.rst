===================
'Report': Graphical Results
===================

Introduction
===================

By ``Report``, user can view the graphical results of the experiment.

There are the following graphics to view:

- analysis_position
    - report_graph
    - score_ic_graph
    - cumulative_return_graph
    - risk_analysis_graph
    - rank_label_graph

- analysis_model
    - model_performance_graph


Example
===================

.. note::

    The following is a simple example of drawing.
    For more features, please see the function document: similar to ``help(qcr.analysis_position.report_graph)``


Get all supported graphics. Please see the API section at the bottom of the page for details:

.. code-block:: python

    >>> import qlib.contrib.report as qcr
    >>> print(qcr.GRAPH_NAME_LISt)
    ['analysis_position.report_graph', 'analysis_position.score_ic_graph', 'analysis_position.cumulative_return_graph', 'analysis_position.risk_analysis_graph', 'analysis_position.rank_label_graph', 'analysis_model.model_performance_graph']





API
===================



.. automodule:: qlib.contrib.report.analysis_position.report
    :members:



.. automodule:: qlib.contrib.report.analysis_position.score_ic
    :members:



.. automodule:: qlib.contrib.report.analysis_position.cumulative_return
    :members:



.. automodule:: qlib.contrib.report.analysis_position.risk_analysis
    :members:



.. automodule:: qlib.contrib.report.analysis_position.rank_label
    :members:



.. automodule:: qlib.contrib.report.analysis_model.analysis_model_performance
    :members:

