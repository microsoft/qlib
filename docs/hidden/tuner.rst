.. _tuner:

Tuner
===================
.. currentmodule:: qlib

Introduction
-------------------

Welcome to use Tuner, this document is based on that you can use Estimator proficiently and correctly.

You can find the optimal hyper-parameters and combinations of models, trainers, strategies and data labels.

The usage of program `tuner` is similar with `estimator`, you need provide the URL of the configuration file.
The `tuner` will do the following things:

- Construct tuner pipeline
- Search and save best hyper-parameters of one tuner
- Search next tuner in pipeline
- Save the global best hyper-parameters and combination

Each tuner is consisted with a kind of combination of modules, and its goal is searching the optimal hyper-parameters of this combination.
The pipeline is consisted with different tuners, it is aim at finding the optimal combination of modules.

The result will be printed on screen and saved in file, you can check the result in your experiment saving files.

Example
~~~~~~~

Let's see an example,

First make sure you have the latest version of `qlib` installed.

Then, you need to privide a configuration to setup the experiment.
We write a simple configuration example as following,

.. code-block:: YAML

    experiment:
        name: tuner_experiment
        tuner_class: QLibTuner
    qlib_client:
        auto_mount: False
        logging_level: INFO 
    optimization_criteria:
        report_type: model
        report_factor: model_score
        optim_type: max
    tuner_pipeline:
      - 
        model: 
            class: SomeModel
            space: SomeModelSpace
        trainer: 
            class: RollingTrainer
        strategy: 
            class: TopkAmountStrategy
            space: TopkAmountStrategySpace
        max_evals: 2

    time_period:
        rolling_period: 360
        train_start_date: 2005-01-01
        train_end_date: 2014-12-31
        validate_start_date: 2015-01-01
        validate_end_date: 2016-06-30
        test_start_date: 2016-07-01
        test_end_date: 2018-04-30
    data:
        class: ALPHA360
        provider_uri: /data/qlib
        args:
            start_date: 2005-01-01
            end_date: 2018-04-30
            dropna_label: True
            dropna_feature: True
        filter:
            market: csi500
            filter_pipeline:
              -
                class: NameDFilter
                module_path: qlib.data.filter
                args:
                  name_rule_re: S(?!Z3)
                  fstart_time: 2018-01-01
                  fend_time: 2018-12-11
              -
                class: ExpressionDFilter
                module_path: qlib.data.filter
                args:
                  rule_expression: $open/$factor<=45
                  fstart_time: 2018-01-01
                  fend_time: 2018-12-11
    backtest:
        normal_backtest_args:
            verbose: False
            limit_threshold: 0.095
            account: 500000
            benchmark: SH000905
            deal_price: vwap
        long_short_backtest_args:
            topk: 50

Next, we run the following command, and you can see:

.. code-block:: bash

    ~/v-yindzh/Qlib/cfg$ tuner -c tuner_config.yaml

    Searching params: {'model_space': {'colsample_bytree': 0.8870905643607678, 'lambda_l1': 472.3188735122233, 'lambda_l2': 92.75390994877243, 'learning_rate': 0.09741751430635413, 'loss': 'mse', 'max_depth': 8, 'num_leaves': 160, 'num_threads': 20, 'subsample': 0.7536051584789751}, 'strategy_space': {'buffer_margin': 250, 'topk': 40}}
    ...
    (Estimator experiment screen log)
    ...
    Searching params: {'model_space': {'colsample_bytree': 0.6667379039007301, 'lambda_l1': 382.10698024977904, 'lambda_l2': 117.02506488151757, 'learning_rate': 0.18514539615228137, 'loss': 'mse', 'max_depth': 6, 'num_leaves': 200, 'num_threads': 12, 'subsample': 0.9449255686969292}, 'strategy_space': {'buffer_margin': 200, 'topk': 30}}
    ...
    (Estimator experiment screen log)
    ...
    Local best params: {'model_space': {'colsample_bytree': 0.6667379039007301, 'lambda_l1': 382.10698024977904, 'lambda_l2': 117.02506488151757, 'learning_rate': 0.18514539615228137, 'loss': 'mse', 'max_depth': 6, 'num_leaves': 200, 'num_threads': 12, 'subsample': 0.9449255686969292}, 'strategy_space': {'buffer_margin': 200, 'topk': 30}}
    Time cost: 489.87220 | Finished searching best parameters in Tuner 0.
    Time cost: 0.00069 | Finished saving local best tuner parameters to: tuner_experiment/estimator_experiment/estimator_experiment_0/local_best_params.json .
    Searching params: {'data_label_space': {'labels': ('Ref($vwap, -2)/Ref($vwap, -1) - 2',)}, 'model_space': {'input_dim': 158, 'lr': 0.001, 'lr_decay': 0.9100529502185579, 'lr_decay_steps': 162.48901403763966, 'optimizer': 'gd', 'output_dim': 1}, 'strategy_space': {'buffer_margin': 300, 'topk': 35}}
    ...
    (Estimator experiment screen log)
    ...
    Searching params: {'data_label_space': {'labels': ('Ref($vwap, -2)/Ref($vwap, -1) - 1',)}, 'model_space': {'input_dim': 158, 'lr': 0.1, 'lr_decay': 0.9882802970847494, 'lr_decay_steps': 164.76742865207729, 'optimizer': 'adam', 'output_dim': 1}, 'strategy_space': {'buffer_margin': 250, 'topk': 35}}
    ...
    (Estimator experiment screen log)
    ...
    Local best params: {'data_label_space': {'labels': ('Ref($vwap, -2)/Ref($vwap, -1) - 1',)}, 'model_space': {'input_dim': 158, 'lr': 0.1, 'lr_decay': 0.9882802970847494, 'lr_decay_steps': 164.76742865207729, 'optimizer': 'adam', 'output_dim': 1}, 'strategy_space': {'buffer_margin': 250, 'topk': 35}}
    Time cost: 550.74039 | Finished searching best parameters in Tuner 1.
    Time cost: 0.00023 | Finished saving local best tuner parameters to: tuner_experiment/estimator_experiment/estimator_experiment_1/local_best_params.json .
    Time cost: 1784.14691 | Finished tuner pipeline.
    Time cost: 0.00014 | Finished save global best tuner parameters.
    Best Tuner id: 0.
    You can check the best parameters at tuner_experiment/global_best_params.json.


Finally, you can check the results of your experiment in the given path.

Configuration file
------------------

Before using `tuner`, you need to prepare a configuration file. Next we will show you how to prepare each part of the configuration file.

About the experiment
~~~~~~~~~~~~~~~~~~~~

First, your configuration file needs to have a field about the experiment, whose key is `experiment`, this field and its contents determine the saving path and tuner class.

Usually it should contain the following content:

.. code-block:: YAML

    experiment:
        name: tuner_experiment
        tuner_class: QLibTuner

Also, there are some optional fields. The meaning of each field is as follows:

- `name`
    The experiment name, str type, the program will use this experiment name to construct a directory to save the process of the whole experiment and the results. The default value is `tuner_experiment`.

- `dir`
    The saving path, str type, the program will construct the experiment directory in this path. The default value is the path where configuration locate.

- `tuner_class`
    The class of tuner, str type, must be an already implemented model, such as `QLibTuner` in `qlib`, or a custom tuner, but it must be a subclass of `qlib.contrib.tuner.Tuner`, the default value is `QLibTuner`.

- `tuner_module_path`
    The module path, str type, absolute url is also supported, indicates the path of the implementation of tuner. The default value is `qlib.contrib.tuner.tuner` 

About the optimization criteria
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You need to designate a factor to optimize, for tuner need a factor to decide which case is better than other cases.
Usually, we use the result of `estimator`, such as backtest results and the score of model. 

This part needs contain these fields:

.. code-block:: YAML

    optimization_criteria:
        report_type: model
        report_factor: model_pearsonr
        optim_type: max

- `report_type`
    The type of the report, str type, determines which kind of report you want to use. If you want to use the backtest result type, you can choose `pred_long`, `pred_long_short`, `pred_short`, `excess_return_without_cost` and `excess_return_with_cost`. If you want to use the model result type, you can only choose `model`.

- `report_factor`
    The factor you want to use in the report, str type, determines which factor you want to optimize. If your `report_type` is backtest result type, you can choose `annualized_return`, `information_ratio`, `max_drawdown`, `mean` and `std`. If your `report_type` is model result type, you can choose `model_score` and `model_pearsonr`.

- `optim_type`
    The optimization type, str type, determines what kind of optimization you want to do. you can minimize the factor or maximize the factor, so you can choose `max`, `min` or `correlation` at this field.
    Note: `correlation` means the factor's best value is 1, such as `model_pearsonr` (a corraltion coefficient).

If you want to process the factor or you want fetch other kinds of factor, you can override the `objective` method in your own tuner.

About the tuner pipeline
~~~~~~~~~~~~~~~~~~~~~~~~

The tuner pipeline contains different tuners, and the `tuner` program will process each tuner in pipeline. Each tuner will get an optimal hyper-parameters of its specific combination of modules. The pipeline will contrast the results of each tuner, and get the best combination and its optimal hyper-parameters. So, you need to configurate the pipeline and each tuner, here is an example:

.. code-block:: YAML

    tuner_pipeline:
      - 
        model: 
            class: SomeModel
            space: SomeModelSpace
        trainer: 
            class: RollingTrainer
        strategy: 
            class: TopkAmountStrategy
            space: TopkAmountStrategySpace
        max_evals: 2

Each part represents a tuner, and its modules which are to be tuned. Space in each part is the hyper-parameters' space of a certain module, you need to create your searching space and modify it in `/qlib/contrib/tuner/space.py`. We use `hyperopt` package to help us to construct the space, you can see the detail of how to use it in https://github.com/hyperopt/hyperopt/wiki/FMin .

- model
    You need to provide the `class` and the `space` of the model. If the model is user's own implementation, you need to privide the `module_path`. 

- trainer
    You need to proveide the `class` of the trainer. If the trainer is user's own implementation, you need to privide the `module_path`. 

- strategy
    You need to provide the `class` and the `space` of the strategy. If the strategy is user's own implementation, you need to privide the `module_path`. 

- data_label
    The label of the data, you can search which kinds of labels will lead to a better result. This part is optional, and you only need to provide `space`.

- max_evals
    Allow up to this many function evaluations in this tuner. The default value is 10.

If you don't want to search some modules, you can fix their spaces in `space.py`. We will not give the default module.

About the time period
~~~~~~~~~~~~~~~~~~~~~

You need to use the same dataset to evaluate your different `estimator` experiments in `tuner` experiment. Two experiments using different dataset are uncomparable. You can specify `time_period` through the configuration file:

.. code-block:: YAML

    time_period:
        rolling_period: 360
        train_start_date: 2005-01-01
        train_end_date: 2014-12-31
        validate_start_date: 2015-01-01
        validate_end_date: 2016-06-30
        test_start_date: 2016-07-01
        test_end_date: 2018-04-30

- `rolling_period`    
    The rolling period, integer type, indicates how many time steps need rolling when rolling the data. The default value is `60`. If you use `RollingTrainer`, this config will be used, or it will be ignored.

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

About the data and backtest
~~~~~~~~~~~~~~~~~~~~~~~~~~~

`data` and `backtest` are all same in the whole `tuner` experiment. Different `estimator` experiments must use the same data and backtest method. So, these two parts of config are same with that in `estimator` configuration. You can see the precise defination of these parts in `estimator` introduction. We only provide an example here.

.. code-block:: YAML

    data:
        class: ALPHA360
        provider_uri: /data/qlib
        args:
            start_date: 2005-01-01
            end_date: 2018-04-30
            dropna_label: True
            dropna_feature: True
            feature_label_config: /home/v-yindzh/v-yindzh/QLib/cfg/feature_config.yaml
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
    backtest:
        normal_backtest_args:
            verbose: False
            limit_threshold: 0.095
            account: 500000
            benchmark: SH000905
            deal_price: vwap
        long_short_backtest_args:
            topk: 50

Experiment Result
-----------------

All the results are stored in experiment file directly, you can check them directly in the corresponding files. 
What we save are as following:

- Global optimal parameters
- Local optimal parameters of each tuner
- Config file of this `tuner` experiment
- Every `estimator` experiments result in the process

