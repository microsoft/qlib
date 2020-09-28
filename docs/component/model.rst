.. _model:
============================================
Interday Model: Model Training & Prediction
============================================

Introduction
===================

``Interday Model`` is designed to make the `prediction score` about stocks. Users can use the ``Interday Model`` in an automatic workflow by ``Estimator``, please refer to `Estimator: Workflow Management <estimator.html>`_.  

Because the components in ``Qlib`` are designed in a loosely-coupled way, ``Interday Model`` can be used as an independent module also.

Base Class & Interface
======================

``Qlib`` provides a base class `qlib.contrib.model.base.Model <../reference/api.html#module-qlib.contrib.model.base>`_ from which all models should inherit.

The base class provides the following interfaces:

- `__init__(**kwargs)`
    - Initialization.
    - If users use ``Estimator`` to start an `experiment`, the parameter of `__init__` method shoule be consistent with the hyperparameters in the configuration file.

- `fit(self, x_train, y_train, x_valid, y_valid, w_train=None, w_valid=None, **kwargs)`
    - Train model.
    - Parameter:
        - `x_train`, pd.DataFrame type, train feature
            The following example explains the value of `x_train`:

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

            
            `x_train` is a pandas DataFrame, whose index is MultiIndex <instrument(str), datetime(pd.Timestamp)>. Each column of `x_train` corresponds to a feature, and the column name is the feature name. 
            
            .. note::
            
                The number and names of the columns are determined by the data handler, please refer to `Data Handler <data.html#data-handler>`_ and `Estimator Data Section <estimator.html#data-section>`_.
            
        - `y_train`, pd.DataFrame type, train label
            The following example explains the value of `y_train`:

             .. code-block:: YAML
                                
                                        LABEL
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
            
            `y_train` is a pandas DataFrame, whose index is MultiIndex <instrument(str), datetime(pd.Timestamp)>. The `LABEL` column represents the value of train label.

            .. note::

                The number and names of the columns are determined by the ``Data Handler``, please refer to `Data Handler <data.html#data-handler>`_.

        - `x_valid`, pd.DataFrame type, validation feature
            The format of `x_valid` is same as `x_train`


        - `y_valid`, pd.DataFrame type, validation label
            The format of `y_valid` is same as `y_train`

        - `w_train`(Optional args, default is None), pd.DataFrame type, train weight
            `w_train` is a pandas DataFrame, whose shape and index is same as `x_train`. The float value in `w_train` represents the weight of the feature at the same position in `x_train`.

        - `w_train`(Optional args, default is None), pd.DataFrame type, validation weight
            `w_train` is a pandas DataFrame, whose shape and index is the same as `x_valid`. The float value in `w_train` represents the weight of the feature at the same position in `x_train`.

- `predict(self, x_test, **kwargs)`
    - Predict test data 'x_test'
    - Parameter:
        - `x_test`, pd.DataFrame type, test features
            The form of `x_test` is same as `x_train` in 'fit' method.
    - Return: 
        - `label`, np.ndarray type, test label
            The label of `x_test` that predicted by model.

- `score(self, x_test, y_test, w_test=None, **kwargs)`
    - Evaluate model with test feature/label
    - Parameter:
        - `x_test`, pd.DataFrame type, test feature
            The format of `x_test` is same as `x_train` in `fit` method.
        
        - `x_test`, pd.DataFrame type, test label
            The format of `y_test` is same as `y_train` in `fit` method.

        - `w_test`, pd.DataFrame type, test weight
            The format of `w_test` is same as `w_train` in `fit` method.
    - Return: float type, evaluation score

For other interfaces such as `save`, `load`, `finetune`, please refer to `Model API <../reference/api.html#module-qlib.contrib.model.base>`_.

Example
==================

``Qlib`` provides ``LightGBM`` and ``DNN`` models as the baseline, the following steps show how to run`` LightGBM`` as an independent module.

- Initialize ``Qlib`` with `qlib.init` first, please refer to `Initialization <../start/initialization.html>`_.
- Run the following code to get the `prediction score` `pred_score`
    .. code-block:: Python

        from qlib.contrib.estimator.handler import Alpha158
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

        x_train, y_train, x_validate, y_validate, x_test, y_test = Alpha158(
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
        model = LGBModel(**MODEL_CONFIG)
        model.fit(x_train, y_train, x_validate, y_validate)
        _pred = model.predict(x_test)
        pred_score = pd.DataFrame(index=_pred.index)
        pred_score["score"] = _pred.iloc(axis=1)[0]

    .. note:: `Alpha158` is the data handler provided by ``Qlib``, please refer to `Data Handler <data.html#data-handler>`_.

Also, the above example has been given in ``examples/train_backtest_analyze.ipynb``.

Custom Model
===================

Qlib supports custom models. If users are interested in customizing their own models and integrating the models into ``Qlib``, please refer to `Custom Model Integration <../start/integration.html>`_.


API
===================
Please refer to `Model API <../reference/api.html#module-qlib.contrib.model.base>`_.
