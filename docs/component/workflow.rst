.. _workflow:

=================================
Workflow: Workflow Management
=================================
.. currentmodule:: qlib

Introduction
===================

The components in `Qlib Framework <../introduction/introduction.html#framework>`_ are designed in a loosely-coupled way. Users could build their own Quant research workflow with these components like `Example <https://github.com/microsoft/qlib/blob/main/examples/workflow_by_code.py>`_.


Besides, ``Qlib`` provides more user-friendly interfaces named ``qrun`` to automatically run the whole workflow defined by configuration. Running the whole workflow is called an `execution`.
With ``qrun``, user can easily start an `execution`, which includes the following steps:

- Data
    - Loading
    - Processing
    - Slicing
- Model
    - Training and inference
    - Saving & loading
- Evaluation
    - Forecast signal analysis
    - Backtest

For each `execution`, ``Qlib`` has a complete system to tracking all the information as well as artifacts generated during training, inference and evaluation phase. For more information about how ``Qlib`` handles this, please refer to the related document: `Recorder: Experiment Management <../component/recorder.html>`_.

Complete Example
===================

Before getting into details, here is a complete example of ``qrun``, which defines the workflow in typical Quant research.
Below is a typical config file of ``qrun``.

.. code-block:: YAML

    qlib_init:
        provider_uri: "~/.qlib/qlib_data/cn_data"
        region: cn
    market: &market csi300
    benchmark: &benchmark SH000300
    data_handler_config: &data_handler_config
        start_time: 2008-01-01
        end_time: 2020-08-01
        fit_start_time: 2008-01-01
        fit_end_time: 2014-12-31
        instruments: *market
    port_analysis_config: &port_analysis_config
        strategy:
            class: TopkDropoutStrategy
            module_path: qlib.contrib.strategy.strategy
            kwargs:
                topk: 50
                n_drop: 5
        backtest:
            verbose: False
            limit_threshold: 0.095
            account: 100000000
            benchmark: *benchmark
            deal_price: close
            open_cost: 0.0005
            close_cost: 0.0015
            min_cost: 5
    task:
        model:
            class: LGBModel
            module_path: qlib.contrib.model.gbdt
            kwargs:
                loss: mse
                colsample_bytree: 0.8879
                learning_rate: 0.0421
                subsample: 0.8789
                lambda_l1: 205.6999
                lambda_l2: 580.9768
                max_depth: 8
                num_leaves: 210
                num_threads: 20
        dataset:
            class: DatasetH
            module_path: qlib.data.dataset
            kwargs:
                handler:
                    class: Alpha158
                    module_path: qlib.contrib.data.handler
                    kwargs: *data_handler_config
                segments:
                    train: [2008-01-01, 2014-12-31]
                    valid: [2015-01-01, 2016-12-31]
                    test: [2017-01-01, 2020-08-01]
        record: 
            - class: SignalRecord
              module_path: qlib.workflow.record_temp
              kwargs: {}
            - class: PortAnaRecord
              module_path: qlib.workflow.record_temp
              kwargs: 
                  config: *port_analysis_config

After saving the config into `configuration.yaml`, users could start the workflow and test their ideas with a single command below.

.. code-block:: bash

    qrun configuration.yaml

If users want to use ``qrun`` under debug mode, please use the following command:

.. code-block:: bash

    python -m pdb qlib/workflow/cli.py examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml

.. note:: 

    `qrun` will be placed in your $PATH directory when installing ``Qlib``.

.. note:: 
        
    The symbol `&` in `yaml` file stands for an anchor of a field, which is useful when another fields include this parameter as part of the value. Taking the configuration file above as an example, users can directly change the value of `market` and `benchmark` without traversing the entire configuration file.


Configuration File
===================

Let's get into details of ``qrun`` in this section.

Before using ``qrun``, users need to prepare a configuration file. The following content shows how to prepare each part of the configuration file.

Qlib Init Section
--------------------

At first, the configuration file needs to contain several basic parameters which will be used for qlib initialization.

.. code-block:: YAML

    provider_uri: "~/.qlib/qlib_data/cn_data"
    region: cn

The meaning of each field is as follows:

- `provider_uri`
    Type: str. The URI of the Qlib data. For example, it could be the location where the data loaded by ``get_data.py`` are stored.

- `region`
    - If `region` == "us", ``Qlib`` will be initialized in US-stock mode. 
    - If `region` == "cn", ``Qlib`` will be initialized in China-stock mode.

    .. note:: 
        
        The value of `region` should be aligned with the data stored in `provider_uri`.


Task Section
--------------------

The `task` field in the configuration corresponds to a `task`, which contains the parameters of three different subsections: `Model`, `Dataset` and `Record`.

Model Section
~~~~~~~~~~~~~~~~~~~~

In the `task` field, the `model` section describes the parameters of the model to be used for training and inference. For more information about the base ``Model`` class, please refer to `Qlib Model <../component/model.html>`_.

.. code-block:: YAML

    model:
        class: LGBModel
        module_path: qlib.contrib.model.gbdt
        kwargs:
            loss: mse
            colsample_bytree: 0.8879
            learning_rate: 0.0421
            subsample: 0.8789
            lambda_l1: 205.6999
            lambda_l2: 580.9768
            max_depth: 8
            num_leaves: 210
            num_threads: 20

The meaning of each field is as follows:

- `class`
    Type: str. The name for the model class.

- `module_path`
    Type: str. The path for the model in qlib.

- `kwargs`
    The keywords arguments for the model. Please refer to the specific model implementation for more information: `models <https://github.com/microsoft/qlib/blob/main/qlib/contrib/model>`_. 

.. note:: 
        
    ``Qlib`` provides a util named: ``init_instance_by_config`` to initialize any class inside ``Qlib`` with the configuration includes the fields: `class`, `module_path` and `kwargs`.

Dataset Section
~~~~~~~~~~~~~~~~~~~~

The `dataset` field describes the parameters for the ``Dataset`` module in ``Qlib`` as well those for the module ``DataHandler``. For more information about the ``Dataset`` module, please refer to `Qlib Model <../component/data.html#dataset>`_.

The keywords arguments configuration of the ``DataHandler`` is as follows:

.. code-block:: YAML

    data_handler_config: &data_handler_config
        start_time: 2008-01-01
        end_time: 2020-08-01
        fit_start_time: 2008-01-01
        fit_end_time: 2014-12-31
        instruments: *market

Users can refer to the document of `DataHandler <../component/data.html#datahandler>`_ for more information about the meaning of each field in the configuration.

Here is the configuration for the ``Dataset`` module which will take care of data preprossing and slicing during the training and testing phase.

.. code-block:: YAML

    dataset:
        class: DatasetH
        module_path: qlib.data.dataset
        kwargs:
            handler:
                class: Alpha158
                module_path: qlib.contrib.data.handler
                kwargs: *data_handler_config
            segments:
                train: [2008-01-01, 2014-12-31]
                valid: [2015-01-01, 2016-12-31]
                test: [2017-01-01, 2020-08-01]

Record Section
~~~~~~~~~~~~~~~~~~~~

The `record` field is about the parameters the ``Record`` module in ``Qlib``. ``Record`` is responsible for tracking training process and results such as `information Coefficient (IC)` and `backtest` in a standard format.

The following script is the configuration of `backtest` and the `strategy` used in `backtest`:

.. code-block:: YAML

    port_analysis_config: &port_analysis_config
        strategy:
            class: TopkDropoutStrategy
            module_path: qlib.contrib.strategy.strategy
            kwargs:
                topk: 50
                n_drop: 5
        backtest:
            verbose: False
            limit_threshold: 0.095
            account: 100000000
            benchmark: *benchmark
            deal_price: close
            open_cost: 0.0005
            close_cost: 0.0015
            min_cost: 5

For more information about the meaning of each field in configuration of `strategy` and `backtest`, users can look up the documents: `Strategy <../component/strategy.html>`_ and `Backtest <../component/backtest.html>`_.

Here is the configuration details of different `Record Template` such as ``SignalRecord`` and ``PortAnaRecord``:

.. code-block:: YAML

    record: 
        - class: SignalRecord
          module_path: qlib.workflow.record_temp
          kwargs: {}
        - class: PortAnaRecord
          module_path: qlib.workflow.record_temp
          kwargs: 
            config: *port_analysis_config

For more information about the ``Record`` module in ``Qlib``, user can refer to the related document: `Record <../component/recorder.html#record-template>`_.
