.. _estimator:
=================================
Estimator: Workflow Management
=================================
.. currentmodule:: qlib

Introduction
===================

The components in `Qlib Framework <../introduction/introduction.html#framework>`_ are designed in a loosely-coupled way. Users could build their own Quant research workflow with these components like `Example <https://github.com/microsoft/qlib/blob/main/examples/train_and_backtest.py>`_


Besides, ``Qlib`` provides more user-friendly interfaces named ``Estimator`` to automatically run the whole workflow defined by configuration.  A concrete execution of the whole workflow is called an `experiment`.
With ``Estimator``, user can easily run an `experiment`, which includes the following steps:

- Data
    - Loading
    - Processing
    - Slicing
- Model
    - Training and inference(static or rolling)
    - Saving & loading
- Evaluation(Back-testing)

For each `experiment`, ``Qlib`` will capture the model training details, performance evaluation results and basic information (e.g. names, ids). The captured data will be stored in backend-storage (disk or database).

Complete Example
===================

Before getting into details, here is a complete example of ``Estimator``, which defines the workflow in typical Quant research.
Below is a typical config file of ``Estimator``.

.. code-block:: YAML

    experiment:
      name: estimator_example
      observer_type: file_storage
      mode: train
    model:
      class: LGBModel
      module_path: qlib.contrib.model.gbdt
      args:
        loss: mse
        colsample_bytree: 0.8879
        learning_rate: 0.0421
        subsample: 0.8789
        lambda_l1: 205.6999
        lambda_l2: 580.9768
        max_depth: 8
        num_leaves: 210
        num_threads: 20
    data:
      class: Alpha158
      args:
        dropna_label: True
      filter:
        market: csi500
    trainer:
      class: StaticTrainer
      args:
        rolling_period: 360
        train_start_date: 2007-01-01
        train_end_date: 2014-12-31
        validate_start_date: 2015-01-01
        validate_end_date: 2016-12-31
        test_start_date: 2017-01-01
        test_end_date: 2020-08-01
    strategy:
      class: TopkDropoutStrategy
      args:
        topk: 50
        n_drop: 5
    backtest:
      normal_backtest_args:
        verbose: False
        limit_threshold: 0.095
        account: 100000000
        benchmark: SH000905
        deal_price: close
        open_cost: 0.0005
        close_cost: 0.0015
        min_cost: 5
    qlib_data:
      # when testing, please modify the following parameters according to the specific environment
      provider_uri: "~/.qlib/qlib_data/cn_data"
      region: "cn"

After saving the config into `configuration.yaml`, users could start the workflow and test their ideas with a single command below.

.. code-block:: bash

    estimator -c configuration.yaml

.. note:: `estimator` will be placed in your $PATH directory when installing ``Qlib``.



Configuration File
===================

Let's get into details of ``Estimator`` in this section.

Before using ``estimator``, users need to prepare a configuration file. The following content shows how to prepare each part of the configuration file.

Experiment Section
--------------------

At first, the configuration file needs to contain a section named `experiment` about the basic information. This section describes how `estimator` tracks and persists current `experiment`. ``Qlib`` used `sacred`, a lightweight open-source tool, to configure, organize, generate logs, and manage experiment results. Partial behaviors of `sacred` will base on the `experiment` section.

Following files will be saved by `sacred` after `estimator` finish an `experiment`:

- `model.bin`, model binary file
- `pred.pkl`, model prediction result file
- `analysis.pkl`, backtest performance analysis file
- `positions.pkl`, backtest position records file
- `run`, the experiment information object, usually contains some meta information such as the experiment name, experiment date, etc.

Here is the typical configuration of `experiment section`

.. code-block:: YAML

     experiment:
        name: test_experiment
        observer_type: mongo
        mongo_url: mongodb://MONGO_URL
        db_name: public
        finetune: false
        exp_info_path: /home/test_user/exp_info.json
        mode: test
        loader:
            id: 677
	

The meaning of each field is as follows:

- `name`   
    The experiment name, str type, `sacred <https://github.com/IDSIA/sacred>_` will use this experiment name as an identifier for some important internal processes. Users can find this field in `run` object of `sacred`.  The default value is `test_experiment`.

- `observer_type`
    Observer type, str type, there are two choices which include `file_storage` and `mongo` respectively. If `file_storage` is selected, all the above-mentioned managed contents will be stored in the `dir` directory, separated by the number of times of experiments as a subfolder. If it is `mongo`, the content will be stored in the database. The default is `file_storage`.

    - For `file_storage` observer.
        - `dir`
            Directory URL, str type, directory for `file_storage` observer type, files captured and managed by sacred with `file_storage` observer will be saved to this directory, which is the same directory as `config.json` by default.

    - For `mongo` observer.
        - `mongo_url`
            Database URL, str type, required if the observer type is `mongo`.

        - `db_name`    
            Database name, str type, required if the observer type is `mongo`.

- `finetune`
    ``Estimator``'s behaviors to train models will base on this flag.
    If you just want to train models from scratch each time instead of based on existing models, please leave `finetune=false`. Otherwise please read the
    details below.

    The following table is the processing logic for different situations.

    ==========  ===========================================   ====================================    ===========================================  ==========================================
      .            Static                                                                             Rolling
      .            finetune:true                              finetune:false                          finetune:true                                finetune:false
    ==========  ===========================================   ====================================    ===========================================  ==========================================
    Train       - Need to provide model (Static or Rolling)   - No need to provide model              - Need to provide model (Static or Rolling)  - Need to provide model (Static or Rolling)
                - The args in model section will be           - The args in model section will be     - The args in model section will be          - The args in model section will be
                  used for finetuning                           used for training                       used for finetuning                          used for finetuning
                - Update based on the provided model          - Train model from scratch              - Update based on the provided model         - Based on the provided model update
                  and parameters                                                                        and parameters                             - Train model from scratch
                                                                                                      - **Each rolling time slice is based on**    - **Train each rolling time slice**
                                                                                                        **a model updated from the previous**        **separately**
                                                                                                        **time**        
    Test        - Model must exist, otherwise an exception will be raised.
                - For `StaticTrainer`, users need to train a model and record 'exp_info' for 'Test'.
                - For `RollingTrainer`, users need to train a set of models until the latest time, and record 'exp_info' for 'Test'.
    ==========  =============================================================================================================================================================================

    .. note::

        1. finetune parameters: share model.args parameters.

        2. provide model: from `loader.model_index`, load the index of the model(starting from 0).

        3. If `loader.model_index` is None:
            - In 'Static Finetune=True', if provide 'Rolling', use the last model to update.

            - For `RollingTrainer` with Finetune=True.

                - If `StaticTrainer` is used in loader, the model will be used for initialization for finetuning.

                - If `RollingTrainer` is used in loader, the existing models will be used without any modification and the new models will be initialized with the model in the last period and finetune one by one.


- `exp_info_path`
    save path of experiment info, str type, save the experiment info and model `prediction score` after the experiment is finished. Optional parameter, the default value is `<config_file_dir>/ex_name/exp_info.json`.

- `mode`
    `train` or `test`, str type.
        - `test mode` is designed for inference. Under `test mode`, it will load the model according to the parameters of `loader` and skip model training.
        - `train model`  is the default value. It will train new models by default and 
    Please note that when it fails to load model, it will fall back to `fit` model.
    
    .. note::

        if users choose ` test mode`, they need to make sure:
        - The loader of `test_start_date` must be less than or equal to the current `test_start_date`.
        - If other parameters of the `loader` model args are different, a warning will appear.


- `loader`
    If you just want to train models from scratch each time instead of based on existing models, please ignore `loader` section. Otherwise please read the
    details below.

    The `loader` section only works when the `mode` is `test` or `finetune` is `true`.

    - `model_index`
        Model index, int type. The index of the loaded model in loader_models (starting at 0) for the first `finetune`. The default value is None.

    - `exp_info_path`
        Loader model experiment info path, str type. If the field exists, the following parameters will be parsed from `exp_info_path`, and the following parameters will not work. One of this field and `id` must exist at least .

    - `id`
        The experiment id of the model that needs to be loaded, int type. If the `mode` is `test`, this value is required. This field and `exp_info_path` must exist one.

    - `name`
        The experiment name of the model that needs to be loaded, str type. The default value is the current experiment `name`.

    - `observer_type`
        The experiment observer type of the model that needs to be loaded, str type. The default value is the current experiment `observer_type`.
	
        .. note:: The observer type is a concept of the `sacred` module, which determines how files, standard input, and output which are managed by sacred are stored.
        
        
        - `file_storage`
            If `observer_type` is `file_storage`, the config may be as follows.

            .. code-block:: YAML

                experiment:
                    name: test_experiment
                    dir: <path to a directory> # default is dir of `config.yml`
                    observer_type: file_storage
        - `mongo`
            If `observer_type` is `mongo`, the config may be as follows.

            .. code-block:: YAML

                experiment:
                    name: test_experiment
                    observer_type: mongo
                    mongo_url: mongodb://MONGO_URL
                    db_name: public

            Users need to indicate `mongo_url` and `db_name` for a mongo observer.
            
            .. note::

                If users choose the mongo observer, they need to make sure:
                    - Have an environment with the mongodb installed and a mongo database dedicated to storing the results of the experiments.
                    - The python environment (the version of python and package) to run the experiments and the one to fetch the results are consistent.

Model Section
-----------------

Users can use a specified model by configuration with hyper-parameters.

Custom Models
~~~~~~~~~~~~~~~~~

Qlib supports custom models, but it must be a subclass of the `qlib.contrib.model.Model`, the config for a custom model may be as following.

.. code-block:: YAML

    model:
        class: SomeModel
        module_path: /tmp/my_experment/custom_model.py
        args:
            loss: binary


The class `SomeModel` should be in the module `custom_model`, and ``Qlib`` could parse the `module_path` to load the class.

To know more about ``Interday Model``, please refer to `Interday Model: Training & Prediction <model.html>`_.

Data Section
-----------------

``Data Handler`` can be used to load raw data, prepare features and label columns, preprocess data (standardization, remove NaN, etc.), split training, validation, and test sets. It is a subclass of `qlib.contrib.estimator.handler.BaseDataHandler`.

Users can use the specified data handler by config as follows.

.. code-block:: YAML

    data:
        class: Alpha158
        args:
            start_date: 2005-01-01
            end_date: 2018-04-30  
            dropna_label: True
        filter:
            market: csi500
            filter_pipeline:
              -
                class: NameDFilter
                module_path: qlib.filter
                args:
                  name_rule_re: S(?!Z3)
                  fstart_time: 2018-01-01
                  fend_time: 2018-12-11
              -
                class: ExpressionDFilter
                module_path: qlib.filter
                args:
                  rule_expression: $open/$factor<=45
                  fstart_time: 2018-01-01
                  fend_time: 2018-12-11

- `class`    
    Data handler class, str type, which should be a subclass of `qlib.contrib.estimator.handler.BaseDataHandler`, and implements 5 important interfaces for loading features, loading raw data, preprocessing raw data, slicing train, validation, and test data. The default value is `ALPHA360`. If users want to write a data handler to retrieve the data in ``Qlib``, `QlibDataHandler` is suggested.

- `module_path`    
   The module path, str type, absolute url is also supported, indicates the path of the `class` implementation of the data processor class. The default value is `qlib.contrib.estimator.handler`.

- `args`
    Parameters used for ``Data Handler`` initialization.

    - `train_start_date`
        Training start time, str type, the default value is `2005-01-01`.

    - `start_date`
        Data start date, str type. 

    - `end_date`
        Data end date, str type. the data from start_date to end_date decides which part of data will be loaded in `datahandler`, users can only use these data in the following parts.

    - `dropna_feature` (Optional in args)
        Drop Nan feature, bool type, the default value is False. 

    - `dropna_label` (Optional in args)
        Drop Nan label, bool type, the default value is True. Some multi-label tasks will use this.

    - `normalize_method` (Optional in args)
        Normalize data by a given method. str type. ``Qlib`` gives two normalizing methods, `MinMax` and `Std`.
        If users want to build their own method, please override `_process_normalize_feature`.
  
- `filter`
    Dynamically filtering the stocks based on the filter pipeline.

    - `market`
        index name, str type, the default value is `csi500`.

    - `filter_pipeline`
        Filter rule list, list type, the default value is []. Can be customized according to users' needs.

        - `class`
            Filter class name, str type.

        - `module_path`
            The module path, str type.

        - `args`
            The filter class parameters, these parameters are set according to the `class`, and all the parameters as kwargs to `class`.

Custom Data Handler
~~~~~~~~~~~~~~~~~~~~~~

Qlib support custom data handler, but it must be a subclass of the ``qlib.contrib.estimator.handler.BaseDataHandler``, the config for custom data handler may be as follows.

.. code-block:: YAML

    data:
        class: SomeDataHandler
        module_path: /tmp/my_experment/custom_data_handler.py
        args:
            start_date: 2005-01-01
            end_date: 2018-04-30  

The class `SomeDataHandler` should be in the module `custom_data_handler`, and ``Qlib`` could parse the `module_path` to load the class.

If users want to load features and labels by config, they can inherit ``qlib.contrib.estimator.handler.ConfigDataHandler``, ``Qlib`` also has provided some preprocess methods in this subclass.
If users want to use qlib data, `QLibDataHandler` is recommended, from which users can inherit the custom class. `QLibDataHandler` is also a subclass of `ConfigDataHandler`.

To know more about ``Data Handler``, please refer to `Data Framework&Usage <data.html>`_.

Trainer Section
-----------------

Users can specify the trainer ``Trainer`` by the config file, which is a subclass of ``qlib.contrib.estimator.trainer.BaseTrainer`` and implement three important interfaces for training the model, restoring the model, and getting model predictions as follows.

- `train`    
    Implement this interface to train the model.

- `load`   
    Implement this interface to recover the model from disk.

- `get_pred`   
    Implement this interface to get model prediction results.

Qlib have provided two implemented trainer,

- `StaticTrainer`   
    The static trainer will be trained using the training, validation, and test data of the data processor static slicing.

- `RollingTrainer`    
    The rolling trainer will use the rolling iterator of the data processor to split data for rolling training.


Users can specify `trainer` with the configuration file:

.. code-block:: YAML

    trainer:
        class: StaticTrainer # or RollingTrainer
        args:
            rolling_period: 360
            train_start_date: 2005-01-01
            train_end_date: 2014-12-31
            validate_start_date: 2015-01-01 
            validate_end_date: 2016-06-30
            test_start_date: 2016-07-01
            test_end_date: 2017-07-31

- `class`   
    Trainer class, which should be a subclass of `qlib.contrib.estimator.trainer.BaseTrainer`, and needs to implement three important interfaces, the default value is `StaticTrainer`.

- `module_path`    
    The module path, str type, absolute url is also supported, indicates the path of the trainer class implementation.

- `args`
    Parameters used for ``Trainer`` initialization.

    - `rolling_period`    
        The rolling period, integer type, indicates how many time steps need rolling when rolling the data. The default value is `60`. Only used in `RollingTrainer`.

    - `train_start_date`
        Training start time, str type.

    - `train_end_date`      
        Training end time, str type.

    - `validate_start_date`    
        Validation start time, str type.

    - `validate_end_date`    
        Validation end time, str type.

    - `test_start_date`    
        Test start time, str type.

    - `test_end_date`     
        Test end time, str type. If `test_end_date` is `-1` or greater than the last date of the data, the last date of the data will be used as `test_end_date`.

Custom Trainer
~~~~~~~~~~~~~~~~~~

Qlib supports custom trainer, but it must be a subclass of the `qlib.contrib.estimator.trainer.BaseTrainer`, the config for a custom trainer may be as following:

.. code-block:: YAML

    trainer:
        class: SomeTrainer
        module_path: /tmp/my_experment/custom_trainer.py
        args:
            train_start_date: 2005-01-01
            train_end_date: 2014-12-31
            validate_start_date: 2015-01-01
            validate_end_date: 2016-06-30
            test_start_date: 2016-07-01
            test_end_date: 2017-07-31


The class `SomeTrainer` should be in the module `custom_trainer`, and ``Qlib`` could parse the `module_path` to load the class.

Strategy Section
-----------------

Users can specify strategy through a config file, for example:

.. code-block:: YAML

    strategy :
        class: TopkDropoutStrategy
        args:
            topk: 50
            n_drop: 5

- `class`
    The strategy class, str type, should be a subclass of `qlib.contrib.strategy.strategy.BaseStrategy`. The default value is `TopkDropoutStrategy`.

- `module_path`
    The module location, str type, absolute url is also supported, and absolute path is also supported, indicates the location of the policy class implementation.

- `args`
    Parameters used for ``Trainer`` initialization.

    - `topk`    
        The number of stocks in the portfolio

    - `n_drop`    
        Number of stocks to be replaced in each trading date

Custom Strategy
^^^^^^^^^^^^^^^^^^^

Qlib supports custom strategy, but it must be a subclass of the ``qlib.contrib.strategy.strategy.BaseStrategy``, the config for custom strategy may be as following:


.. code-block:: YAML

    strategy :
        class: SomeStrategy
        module_path: /tmp/my_experment/custom_strategy.py

The class `SomeStrategy` should be in the module `custom_strategy`, and ``Qlib`` could parse the `module_path` to load the class.

To know more about ``Strategy``, please refer to `Strategy <strategy.html>`_.

Backtest Section
-----------------

Users can specify `backtest` through a config file, for example:

.. code-block:: YAML

    backtest :
        normal_backtest_args:
            topk: 50
            benchmark: SH000905
            account: 500000
            deal_price: close
            min_cost: 5
            subscribe_fields:
              - $close
              - $change
              - $factor

- `normal_backtest_args`
    Normal backtest parameters. All the parameters in this section will be passed to the ``qlib.contrib.evaluate.backtest`` function in the form of `**kwargs`.

    - `benchmark`
        Stock index symbol, str, or list type, the default value is `None`.

        .. note::

            * If `benchmark` is None, it will use the average change of the day of all stocks in 'pred' as the 'bench'.

            * If `benchmark` is list, it will use the daily average change of the stock pool in the list as the 'bench'.

            * If `benchmark` is str, it will use the daily change as the 'bench'.


    - `account`
        Backtest initial cash, integer type. The `account` in `strategy` section is deprecated. It only works when `account` is not set in `backtest` section. It will be overridden by `account` in the `backtest` section. The default value is 1e9.

    - `deal_price`
        Order transaction price field, str type, the default value is close.

    - `min_cost`
        Min transaction cost, float type, the default value is 5.

    - `subscribe_fields`
        Subscribe quote fields, array type, the default value is [`deal_price`, $close, $change, $factor].


Qlib Data Section
--------------------

The `qlib_data` field describes the parameters of qlib initialization.

.. code-block:: YAML

    qlib_data:
      # when testing, please modify the following parameters according to the specific environment
      provider_uri: "~/.qlib/qlib_data/cn_data"
      region: "cn"
    
- `provider_uri`
    The local directory where the data loaded by 'get_data.py' is stored.
- `region`
    - If region == ``qlib.config.REG_CN``, 'qlib' will be initialized in US-stock mode. 
    - If region == ``qlib.config.REG_US``, 'qlib' will be initialized in china-stock mode.

Please refer to `Initialization <../start/initialization.html>`_.

Experiment Result
===================

Form of Experimental Result
----------------------------
The result of the experiment is also the result of the ``Intraday Trading(Backtest)``, please refer to `Intraday Trading: Model&Strategy Testing <backtest.html>`_.


Get Experiment Result
----------------------------

Base Class & Interface
~~~~~~~~~~~~~~~~~~~~~~~

Users can check the experiment results from file storage directly, or check the experiment results from the database, or get the experiment results through two interfaces of a base class `Fetcher` provided by ``Qlib``.

The `Fetcher` provides the following interface
    - `get_experiments(self, exp_name=None):`   
        The interface takes one parameters. The `exp_name` is the experiment name, the default is all experiments. Users can get the returned dictionary with a list of ids and test end date as follows.

        .. code-block:: JSON

            {
                "ex_a": [
                    {
                        "id": 1,
                        "test_end_date": "2017-01-01"
                    }
                ],
                "ex_b": [
                    ...
                ]
            }


    - `get_experiment(exp_name, exp_id, fields=None)`
        The interface takes three parameters. The first parameter is the experiment name, the second parameter is the experiment id, and the third parameter is a list of fields. The default value of `fields` is None, which means all fields.
        

        .. note::
            Currently supported fields:
                ['model', 'analysis', 'positions', 'report_normal', 'pred', 'task_config', 'label']

        Users can get the returned dictionary as follows.

        .. code-block:: JSON

            {
                'analysis': analysis_df,
                'pred': pred_df,
                'positions': positions_dic,
                'report_normal': report_normal_df,
            }

Implemented `Fetcher` s & Examples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Qlib`` provides two implemented `Fetcher` s as follows.

`FileFetcher`
^^^^^^^^^^^^^^^

The `FileFetcher` is a subclass of `Fetcher`, which could fetch files from `file_storage` observer. The following is an example:
.. code-block:: python

    >>> from qlib.contrib.estimator.fetcher import FileFetcher
    >>> f = FileFetcher(experiments_dir=r'./')
    >>> print(f.get_experiments())
    {
        'test_experiment': [
            {
                'id': '1',
                'config': ...
            }, 
            {   
                'id': '2',
                'config': ...
            }, 
            {   
                'id': '3', 
                'config': ...
            }
        ]
    }
    >>> print(f.get_experiment('test_experiment', '1'))
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



`MongoFetcher`
^^^^^^^^^^^^^^^

The `FileFetcher` is a subclass of `Fetcher`, which could fetch files from `mongo` observer. Users should initialize the fetcher with `mongo_url`. The following is an example:

.. code-block:: python

    >>> from qlib.contrib.estimator.fetcher import MongoFetcher
    >>> f = MongoFetcher(mongo_url=..., db_name=...)

