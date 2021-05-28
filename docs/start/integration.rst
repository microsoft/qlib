=========================================
Custom Model Integration
=========================================

Introduction
===================

``Qlib``'s `Model Zoo` includes models such as ``LightGBM``, ``MLP``, ``LSTM``, etc.. These models are examples of ``Forecast Model``. In addition to the default models ``Qlib`` provide, users can integrate their own custom models into ``Qlib``.

Users can integrate their own custom models according to the following steps.

- Define a custom model class, which should be a subclass of the `qlib.model.base.Model <../reference/api.html#module-qlib.model.base>`_.
- Write a configuration file that describes the path and parameters of the custom model.
- Test the custom model.

Custom Model Class
===========================
The Custom models need to inherit `qlib.model.base.Model <../reference/api.html#module-qlib.model.base>`_ and override the methods in it.

- Override the `__init__` method
    - ``Qlib`` passes the initialized parameters to the \_\_init\_\_ method.
    - The hyperparameters of model in the configuration must be consistent with those defined in the `__init__` method.
    - Code Example: In the following example, the hyperparameters of model in the configuration file should contain parameters such as `loss:mse`.
    .. code-block:: Python

        def __init__(self, loss='mse', **kwargs):
            if loss not in {'mse', 'binary'}:
                raise NotImplementedError
            self._scorer = mean_squared_error if loss == 'mse' else roc_auc_score
            self._params.update(objective=loss, **kwargs)
            self._model = None

- Override the `fit` method
    - ``Qlib`` calls the fit method to train the model.
    - The parameters must include training feature `dataset`, which is designed in the interface.
    - The parameters could include some `optional` parameters with default values, such as `num_boost_round = 1000` for `GBDT`.
    - Code Example: In the following example, `num_boost_round = 1000` is an optional parameter.
    .. code-block:: Python
    
        def fit(self, dataset: DatasetH, num_boost_round = 1000, **kwargs):

            # prepare dataset for lgb training and evaluation
            df_train, df_valid = dataset.prepare(
                ["train", "valid"], col_set=["feature", "label"], data_key=DataHandlerLP.DK_L
            )
            x_train, y_train = df_train["feature"], df_train["label"]
            x_valid, y_valid = df_valid["feature"], df_valid["label"]

            # Lightgbm need 1D array as its label
            if y_train.values.ndim == 2 and y_train.values.shape[1] == 1:
                y_train, y_valid = np.squeeze(y_train.values), np.squeeze(y_valid.values)
            else:
                raise ValueError("LightGBM doesn't support multi-label training")

            dtrain = lgb.Dataset(x_train.values, label=y_train)
            dvalid = lgb.Dataset(x_valid.values, label=y_valid)

            # fit the model
            self.model = lgb.train(
                self.params,
                dtrain,
                num_boost_round=num_boost_round,
                valid_sets=[dtrain, dvalid],
                valid_names=["train", "valid"],
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=verbose_eval,
                evals_result=evals_result,
                **kwargs
            )

- Override the `predict` method
    - The parameters must include the parameter `dataset`, which will be userd to get the test dataset.
    - Return the `prediction score`.
    - Please refer to `Model API <../reference/api.html#module-qlib.model.base>`_ for the parameter types of the fit method.
    - Code Example: In the following example, users need to use `LightGBM` to predict the label(such as `preds`) of test data `x_test` and return it.
    .. code-block:: Python

        def predict(self, dataset: DatasetH, **kwargs)-> pandas.Series:
            if self.model is None:
                raise ValueError("model is not fitted yet!")
            x_test = dataset.prepare("test", col_set="feature", data_key=DataHandlerLP.DK_I)
            return pd.Series(self.model.predict(x_test.values), index=x_test.index)

- Override the `finetune` method (Optional)
    - This method is optional to the users. When users want to use this method on their own models, they should inherit the ``ModelFT`` base class, which includes the interface of `finetune`.
    - The parameters must include the parameter `dataset`.
    - Code Example: In the following example, users will use `LightGBM` as the model and finetune it.
    .. code-block:: Python

        def finetune(self, dataset: DatasetH, num_boost_round=10, verbose_eval=20):
            # Based on existing model and finetune by train more rounds
            dtrain, _ = self._prepare_data(dataset)
            self.model = lgb.train(
                self.params,
                dtrain,
                num_boost_round=num_boost_round,
                init_model=self.model,
                valid_sets=[dtrain],
                valid_names=["train"],
                verbose_eval=verbose_eval,
            )

Configuration File
=======================

The configuration file is described in detail in the `Workflow <../component/workflow.html#complete-example>`_ document. In order to integrate the custom model into ``Qlib``, users need to modify the "model" field in the configuration file. The configuration describes which models to use and how we can initialize it.

- Example: The following example describes the `model` field of configuration file about the custom lightgbm model mentioned above, where `module_path` is the module path, `class` is the class name, and `args` is the hyperparameter passed into the __init__ method. All parameters in the field is passed to `self._params` by `\*\*kwargs` in `__init__` except `loss = mse`. 

.. code-block:: YAML
    
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

Users could find configuration file of the baselines of the ``Model`` in ``examples/benchmarks``. All the configurations of different models are listed under the corresponding model folder.

Model Testing
=====================
Assuming that the configuration file is ``examples/benchmarks/LightGBM/workflow_config_lightgbm.yaml``, users can run the following command to test the custom model:

.. code-block:: bash

    cd examples  # Avoid running program under the directory contains `qlib`
    qrun benchmarks/LightGBM/workflow_config_lightgbm.yaml

.. note:: ``qrun`` is a built-in command of ``Qlib``.

Also, ``Model`` can also be tested as a single module. An example has been given in ``examples/workflow_by_code.ipynb``. 


Reference
=====================

To know more about ``Forecast Model``, please refer to `Forecast Model: Model Training & Prediction <../component/model.html>`_ and `Model API <../reference/api.html#module-qlib.model.base>`_.
