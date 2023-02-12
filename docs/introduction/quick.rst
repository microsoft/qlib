
===========
Quick Start
===========

Introduction
============

This ``Quick Start`` guide tries to demonstrate

- It's very easy to build a complete Quant research workflow and try users' ideas with ``Qlib``.
- Though with public data and simple models, machine learning technologies work very well in practical Quant investment.



Installation
============

Users can easily intsall ``Qlib`` according to the following steps:

- Before installing ``Qlib`` from source, users need to install some dependencies:

    .. code-block::

        pip install numpy
        pip install --upgrade  cython

- Clone the repository and install ``Qlib``

    .. code-block::

        git clone https://github.com/microsoft/qlib.git && cd qlib
        python setup.py install

To known more about `installation`, please refer to `Qlib Installation <../start/installation.html>`_.

Prepare Data
============

Load and prepare data by running the following code:

.. code-block::

    python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn

This dataset is created by public data collected by crawler scripts in ``scripts/data_collector/``, which have been released in the same repository. Users could create the same dataset with it.

To known more about `prepare data`, please refer to `Data Preparation <../component/data.html#data-preparation>`_.

Auto Quant Research Workflow
============================

``Qlib`` provides a tool named ``qrun`` to run the whole workflow automatically (including building dataset, training models, backtest and evaluation). Users can start an auto quant research workflow and have a graphical reports analysis according to the following steps:

- Quant Research Workflow:
    - Run  ``qrun`` with a config file of the LightGBM model `workflow_config_lightgbm.yaml` as following.

        .. code-block::

            cd examples  # Avoid running program under the directory contains `qlib`
            qrun benchmarks/LightGBM/workflow_config_lightgbm.yaml


    - Workflow result
        The result of ``qrun`` is as follows, which is also the typical result of ``Forecast model(alpha)``. Please refer to  `Intraday Trading <../component/backtest.html>`_. for more details about the result.

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


    To know more about `workflow` and `qrun`, please refer to `Workflow: Workflow Management <../component/workflow.html>`_.

- Graphical Reports Analysis:
    - Run ``examples/workflow_by_code.ipynb`` with jupyter notebook
        Users can have portfolio analysis or prediction score (model prediction) analysis by run ``examples/workflow_by_code.ipynb``.
    - Graphical Reports
        Users can get graphical reports about the analysis, please refer to `Analysis: Evaluation & Results Analysis <../component/report.html>`_ for more details.



Custom Model Integration
========================

``Qlib`` provides a batch of models (such as ``lightGBM`` and ``MLP`` models) as examples of ``Forecast Model``. In addition to the default model, users can integrate their own custom models into ``Qlib``. If users are interested in the custom model, please refer to `Custom Model Integration <../start/integration.html>`_.
