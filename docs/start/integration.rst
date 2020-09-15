=========================================
Integrate Custom Models into QLib
=========================================

Introduction
===================
The baseline of the qlib model includes lightgbm and dnn. In addition to using the default model, users can their own integrate custom models into qlib.

In order to use the custom model, user can do as follows.

- Define a custom model class, which should be a subclass of the `qlib.contrib.model.base.Model <../reference/api.html#module-qlib.contrib.model.base>`_
- Write a configuration file that describes the path and parameters of the custom model

The following is an example of integrating a custom lightgbm model into qlib:

Define a custom model class
===========================
The Custom models need to inherit `qlib.contrib.model.base.Model <../reference/api.html#module-qlib.contrib.model.base>`_ and override the methods in it.

- Override the `__init__` method
    - Qlib passes the initialized parameters to the \_\_init\_\_ method
    - The parameter must be consistent with the hyperparameters in the configuration file.
    - Code Example: In the following example, the hyperparameter filed of the configuration file should contain parameters such as ‘loss:mse’.
    .. code-block:: Python

        def __init__(self, loss='mse', **kwargs):
            if loss not in {'mse', 'binary'}:
                raise NotImplementedError
            self._scorer = mean_squared_error if loss == 'mse' else roc_auc_score
            self._params.update(objective=loss, **kwargs)
            self._model = None

- Override the `fit` method
    - Qlib calls the fit method to train the model
    - The parameters must include training feature 'x_train', training label 'y_train', test feature 'x_valid', test label 'y_valid'at least.
    - The parameters could include some optional parameters with default values, such as train weight 'w_train', test weight 'w_valid' and 'num_boost_round = 1000'.
    - Code Example: In the following example, 'num_boost_round = 1000' is an optional parameter.
    .. code-block:: Python
    
        def fit(self, x_train:pd.DataFrame, y_train:pd.DataFrame, x_valid:pd.DataFrame, y_valid:pd.DataFrame,
            w_train:pd.DataFrame = None, w_valid:pd.DataFrame = None, num_boost_round = 1000, **kwargs):

            # Lightgbm need 1D array as its label
            if y_train.values.ndim == 2 and y_train.values.shape[1] == 1:
                y_train_1d, y_valid_1d = np.squeeze(y_train.values), np.squeeze(y_valid.values)
            else:
                raise ValueError('LightGBM doesn\'t support multi-label training')

            w_train_weight = None if w_train is None else w_train.values
            w_valid_weight = None if w_valid is None else w_valid.values

            dtrain = lgb.Dataset(x_train.values, label=y_train_1d, weight=w_train_weight)
            dvalid = lgb.Dataset(x_valid.values, label=y_valid_1d, weight=w_valid_weight)
            self._model = lgb.train(
                self._params, 
                dtrain, 
                num_boost_round=num_boost_round,
                valid_sets=[dtrain, dvalid],
                valid_names=['train', 'valid'],
                **kwargs
            )

- Override the `predict` method
    - The parameters include the features of the test data and return the prediction labels
    - Please refer to `qlib.contrib.model.base.Model <../reference/api.html#module-qlib.contrib.model.base>`_ for the parameter types of the fit method
    - Code Example:In the following example, user need to user dnn to predict the label(such as 'preds') of test data 'x_test' and return it.
    .. code-block:: Python

        def predict(self, x_test:pd.DataFrame, **kwargs)-> numpy.ndarray:
            if self._model is None:
                raise ValueError('model is not fitted yet!')
            return self._model.predict(x_test.values)

- Override the `score` method
    - The parameters include the features and label of the test data, and the return loss whose type is passed in the __init__ method
    - Code Example:In the following example, user need to calculate the weighted loss with test data 'x_test',  test label 'y_test' and the weight 'w_test'.
    .. code-block:: Python

        def score(self, x_test:pd.Dataframe, y_test:pd.Dataframe, w_test:pd.DataFrame = None) -> float:
            # Remove rows from x, y and w, which contain Nan in any columns in y_test.
            x_test, y_test, w_test = drop_nan_by_y_index(x_test, y_test, w_test)
            preds = self.predict(x_test)
            w_test_weight = None if w_test is None else w_test.values
            scorer = mean_squared_error if self.loss_type == 'mse' else roc_auc_score
            return scorer(y_test.values, preds, sample_weight=w_test_weight)

- Override the `save` method & `load` method
    - The `save` method parameter include the a `filename` that represents an absolute path, user need to save model into the path.
    - The `load` method parameter include the a `buffer` read from the `filename` passed in `save` method , user need to load model from the `buffer`.
    - Code Example:
    .. code-block:: Python

        def save(self, filename):
            if self._model is None:
                raise ValueError('model is not fitted yet!')
            self._model.save_model(filename)

        def load(self, buffer):
            self._model = lgb.Booster(params={'model_str': buffer.decode('utf-8')})


Write the configuration
=======================

The configuration file is described in detail in the `estimator <../advanced/estimator.html#Example>`_ document. In order to integrate the custom model into qlib, you need to modify the "model" field in the configuration file.

- Example: The following example describes the ‘model’ field of configuration file about the custom lightgbm model mentioned above , where ‘module_path’ is the module path, ‘class’ is the class name, and ‘args’ is the hyperparameter passed into the __init__ method. All parameters in the field is passed to 'self._params' by '\*\*kwargs' in `__init__` except 'loss = mse'. 

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

Test the custom model
=====================
Assuming that the configuration file is named test.yaml, user can run the following command to test the custom model:

.. code-block:: bash

    estimator -c test.yaml

.. note:: 'estimator' is a built-in command of our program.

Also, 'Model' can also be tested as a single module. An example has been given in 'examples.estimator.train_backtest_analyze.ipynb'.

Know More about 'Model'
=====================

If user want to know more about 'model', please refer to document `Use 'Model' to Train&Predict <../advanced/model.rst>`_ and api `qlib.contrib.model.base.Model <../reference/api.html#module-qlib.contrib.model.base>`_.