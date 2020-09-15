===================
Model: Train&Predict
===================

Introduction
===================

By ``Model``, users can use known data and features to train the model and predict the future score of the stock.

Interface
===================

Qlib provides a base class `qlib.contrib.model.base.Model <../reference/api.html#module-qlib.contrib.model.base>`_, which models should inherit from.

The base class provides the following interfaces:

- `def __init__`
    - Initialization.
    - If users use `estimator <../advanced/estimator.html>`_ to start an experiment, the parameter of `__init__` method shoule be consistent with the hyperparameters in the configuration file.

- `def fit(self, x_train, y_train, x_valid, y_valid, w_train=None, w_valid=None, **kwargs)`
    - Train model.
    - Parameter:
        - ``x_train``, pd.DataFrame type, train feature
            The following example explains the value of x_train:

            .. code-block:: YAML
                                
                                        KMID      KLEN      KMID2     KUP       KUP2
                instrument  datetime                                                       
                SH600004    2012-01-04  0.000000  0.017685  0.000000  0.012862  0.727275   
                            2012-01-05 -0.006473  0.025890 -0.250001  0.012945  0.499998   
                            2012-01-06  0.008117  0.019481  0.416666  0.008117  0.416666   
                            2012-01-09  0.016051  0.025682  0.624998  0.006421  0.250001   
                            2012-01-10  0.017323  0.026772  0.647057  0.003150  0.117648   
                ...                         ...       ...       ...       ...       ...   
                SZ300273    2014-12-25 -0.005295  0.038697 -0.136843  0.016293  0.421052   
                            2014-12-26 -0.022486  0.041701 -0.539215  0.002453  0.058824   
                            2014-12-29 -0.031526  0.039092 -0.806451  0.000000  0.000000   
                            2014-12-30 -0.010000  0.032174 -0.310811  0.013913  0.432433   
                            2014-12-31  0.010917  0.020087  0.543479  0.001310  0.065216   

            
            ``x_train`` is a pandas DataFrame, whose index is MultiIndex <instrument(str), datetime(pd.Timestamp)>. Each column of `x_train` corresponds to a feature, and the column name is the feature name. 
            
            .. note::
            
                The number and names of the columns is determined by the data handler, please refer to `Data Handler <data.html#data-handler>`_ and `Estimator Data <estimator.html#about-data>`_.
            
        - ``y_train``, pd.DataFrame type, train label
            The following example explains the value of y_train:

             .. code-block:: YAML
                                
                                        LABEL3
                instrument  datetime            
                SH600004    2012-01-04 -0.798456
                            2012-01-05 -1.366716
                            2012-01-06 -0.491026
                            2012-01-09  0.296900
                            2012-01-10  0.501426
                ...                         ...
                SZ300273    2014-12-25 -0.465540
                            2014-12-26  0.233864
                            2014-12-29  0.471368
                            2014-12-30  0.411914
                            2014-12-31  1.342723
            
            ``y_train`` is a pandas DataFrame, whose index is MultiIndex <instrument(str), datetime(pd.Timestamp)>. The 'LABEL3' column represents the value of train label.

            .. note::

                The number and names of the columns is determined by the data handler, please refer to `Data Handler <data.html#data-handler>`_.

        - ``x_valid``, pd.DataFrame type, validation feature
            The form of ``x_valid`` is same as ``x_train``


        - ``y_valid``, pd.DataFrame type, validation label
            The form of ``y_valid`` is same as ``y_train``

        - ``w_train``(Optional args, default is None), pd.DataFrame type, train weight
            ``w_train`` is a pandas DataFrame, whose shape and index is same as ``x_train``. The float value in ``w_train`` represents the weight of the feature at the same position in ``x_train``.

        - ``w_valid``(Optional args, default is None), pd.DataFrame type, validation weight
            ``w_valid`` is a pandas DataFrame, whose shape and index is same as ``x_valid``. The float value in ``w_train`` represents the weight of the feature at the same position in ``x_train``.

- `def predict(self, x_test, **kwargs)`
    - Predict test data 'x_test'
    - Parameter:
        - ``x_test``, pd.DataFrame type, test features
            The form of ``x_test`` is same as ``x_train`` in 'fit' method.
    - Return: 
        - ``label``, np.ndarray type, test label
            The label of ``x_test`` that predicted by model.

- `def score(self, x_test, y_test, w_test=None, **kwargs)`
    - Evaluate model with test feature/label
    - Parameter:
        - ``x_test``, pd.DataFrame type, test feature
            The form of ``x_test`` is same as ``x_train`` in 'fit' method.
        
        - ``x_test``, pd.DataFrame type, test label
            The form of ``y_test`` is same as ``y_train`` in 'fit' method.

        - ``w_test``, pd.DataFrame type, test weight
            The form of ``w_test`` is same as ``w_train`` in 'fit' method.
    - Return: float type, evaluation score

For other interfaces such as ``save``, ``load``, ``finetune``, please refer to `Model Api <../reference/api.html#module-qlib.contrib.model.base>`_.

Example
==================

'Model' can be run with 'estimator' by modifying the configuration file, and can also be used as a single module. 

Know more about how to run 'Model' with estimator, please refer to `Estimator <estimator.html#about-model>`_.

Qlib provides LightGBM and DNN models as the baseline, the following example shows how to run LightGBM as a single module. 

.. note:: User needs to initialize package qlib with qlib.init first, please refer to `initialization <initialization.rst>`_.


.. code-block:: Python

    from qlib.contrib.estimator.handler import QLibDataHandlerV1
    from qlib.contrib.model.gbdt import LGBModel

    DATA_HANDLER_CONFIG = {
        "dropna_label": True,
        "start_date": "2007-01-01",
        "end_date": "2020-08-01",
        "market": MARKET,
    }

    TRAINER_CONFIG = {
        "train_start_date": "2007-01-01",
        "train_end_date": "2014-12-31",
        "validate_start_date": "2015-01-01",
        "validate_end_date": "2016-12-31",
        "test_start_date": "2017-01-01",
        "test_end_date": "2020-08-01",
    }

    x_train, y_train, x_validate, y_validate, x_test, y_test = QLibDataHandlerV1(
        **DATA_HANDLER_CONFIG
    ).get_split_data(**TRAINER_CONFIG)


    MODEL_CONFIG = {
        "loss": "mse",
        "colsample_bytree": 0.8879,
        "learning_rate": 0.0421,
        "subsample": 0.8789,
        "lambda_l1": 205.6999,
        "lambda_l2": 580.9768,
        "max_depth": 8,
        "num_leaves": 210,
        "num_threads": 20,
    }
    # use default model
    # custom Model, refer to: TODO: Model api url
    model = LGBModel(**MODEL_CONFIG)
    model.fit(x_train, y_train, x_validate, y_validate)
    _pred = model.predict(x_test)

.. note:: 'QLibDataHandlerV1' is the data handler provided by Qlib, please refer to `Data Handler <data.html#data-handler>`_.

Also, the above example has been given in ``examples.estimator.train_backtest_analyze.ipynb``.

Custom Model
===================

Qlib supports custom models, how to customize the model and integrate the model into Qlib, please refer to `How to integrate Model into Qlib <../start/integration.html>`_.


Api
===================
Please refer to `Model Api <../reference/api.html#module-qlib.contrib.model.base>`_ for Model Api.
