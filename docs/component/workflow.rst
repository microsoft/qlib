.. _workflow:

=================================
Workflow: Workflow Management
=================================
.. currentmodule:: qlib

Introduction
===================

The components in `Qlib Framework <../introduction/introduction.html#framework>`_ are designed in a loosely-coupled way. Users could build their own Quant research workflow with these components like `Example <https://github.com/microsoft/qlib/blob/main/examples/workflow_by_code.py>`_.


Besides, ``Qlib`` provides more user-friendly interfaces named ``qrun`` to automatically run the whole workflow defined by configuration.  A concrete execution of the whole workflow is called an `experiment`.
With ``qrun``, user can easily run an `experiment`, which includes the following steps:

- Data
    - Loading
    - Processing
    - Slicing
- Model
    - Training and inference (static or rolling)
    - Saving & loading
- Evaluation
    - Backtest

For each `experiment`, ``Qlib`` has a complete system to tracking all the information as well as artifacts generated during training, inference and evaluation phase. For more information about how Qlib handles `experiment`, please refer to the related document: `Recorder: Experiment Management <../component/recorder.html>`_.

Complete Example
===================

Before getting into details, here is a complete example of ``qrun``, which defines the workflow in typical Quant research.
Below is a typical config file of ``qrun``.

.. code-block:: YAML

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

    qrun -c configuration.yaml

.. note:: 

    `qrun` will be placed in your $PATH directory when installing ``Qlib``.


Configuration File
===================

Let's get into details of ``qrun`` in this section.

Before using ``qrun``, users need to prepare a configuration file. The following content shows how to prepare each part of the configuration file.

Qlib Data Section
--------------------

At first, the configuration file needs to contain several basic parameters about the data, which will be used for qlib initialization, data handling and backtest.

.. code-block:: YAML

    provider_uri: "~/.qlib/qlib_data/cn_data"
    region: cn
    market: &market csi300
    benchmark: &benchmark SH000300

The meaning of each field is as follows:

- `provider_uri`
    Type: str. The URI of the Qlib data. For example, it could be the location where the data loaded by ``get_data.py`` are stored.

- `region`
    - If `region` == "us", ``Qlib`` will be initialized in US-stock mode. 
    - If `region` == "cn", ``Qlib`` will be initialized in china-stock mode.

    .. note:: 
        
        The value of `region` should be aligned with the data stored in `provider_uri`.

- `market`
    Type: str. Index name, the default value is `csi500`.

- `benchmark`
    Type: str, list or pandas.Series. Stock index symbol, the default value is `SH000905`.

    .. note::

        * If `benchmark` is str, it will use the daily change as the 'bench'.

        * If `benchmark` is list, it will use the daily average change of the stock pool in the list as the 'bench'.

        * If `benchmark` is pandas.Series, whose `index` is trading date and the value T is the change from T-1 to T, it will be directly used as the 'bench'. An example is as following:
        
            .. code-block:: python

                print(D.features(D.instruments('csi500'), ['$close/Ref($close, 1)-1'])['$close/Ref($close, 1)-1'].head())
                    2017-01-04    0.011693
                    2017-01-05    0.000721
                    2017-01-06   -0.004322
                    2017-01-09    0.006874
                    2017-01-10   -0.003350
.. note:: 
        
    The symbol `&` in `yaml` file stands for an anchor of a field, which is useful when another fields include this parameter as part of the value. Taking the configuration file above as an example, users can directly change the value of `market` and `benchmark` without traversing the entire configuration file.

Model Section
--------------------

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
--------------------

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
--------------------

The `record` field is about the parameters the ``Record`` module in ``Qlib``. ``Record`` is responsible for generating certain analysis and evaluation results such as `prediction`, `information Coefficient (IC)` and `backtest`.

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