
===============================
Quick Start
===============================

Introduction
==============

This ``Quick Start`` guide tries to demonstrate

- It's very easy to build a complete Quant research workflow and try users' ideas with ``Qlib``.
- Though with public data and simple models, machine learning technologies work very well in practical Quant investment.



Installation
==================

Users can easily intsall ``Qlib`` according to the following steps:

- Before installing ``Qlib`` from source, users need to install some dependencies:

    .. code-block::
        pip install numpy
        pip install --upgrade  cython

- Clone the repository and install ``Qlib``

    .. code-block::

        git clone https://github.com/microsoft/qlib.git && cd qlib
        python setup.py install

To kown more about `installation`, please refer to `Qlib Installation <../start/installation.html>`_.

Prepare Data
==============

Load and prepare data by running the following code:

.. code-block::

    python scripts/get_data.py qlib_data_cn --target_dir ~/.qlib/qlib_data/cn_data

This dataset is created by public data collected by crawler scripts in ``scripts/data_collector/``, which have been released in the same repository. Users could create the same dataset with it.

To kown more about `prepare data`, please refer to `Data Preparation <../component/data.html#data-preparation>`_.

Auto Quant Research Workflow
====================================

``Qlib`` provides a tool named ``Estimator`` to run the whole workflow automatically (including building dataset, training models, backtest and evaluation). Users can start an auto quant research workflow and have a graphical reports analysis according to the following steps: 

- Quant Research Workflow: 
    - Run  ``Estimator`` with `estimator_config.yaml` as following.
        .. code-block:: 

            cd examples  # Avoid running program under the directory contains `qlib`
            estimator -c estimator/estimator_config.yaml


    - Estimator result
        The result of ``Estimator`` is as follows, which is also the result of ``Intraday Trading``. Please refer to  `Intraday Trading <../component/backtest.html>`_. for more details about the result.

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

        
    To know more about `Estimator`, please refer to `Estimator: Workflow Management <../component/estimator.html>`_.

- Graphical Reports Analysis:
    - Run ``examples/estimator/analyze_from_estimator.ipynb`` with jupyter notebook
        Users can have portfolio analysis or prediction score (model prediction) analysis by run ``examples/estimator/analyze_from_estimator.ipynb``.
    - Graphical Reports
        Users can get graphical reports about the analysis, please refer to `Aanalysis: Evaluation & Results Analysis <../component/report.html>`_ for more details.



Custom Model Integration
===============================================

``Qlib`` provides ``lightGBM`` and ``Dnn`` model as the baseline of ``Interday Model``. In addition to the default model, users can integrate their own custom models into ``Qlib``. If users are interested in the custom model, please refer to `Custom Model Integration <../start/integration.html>`_.
