# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# coding=utf-8

from abc import abstractmethod

import pandas as pd
import numpy as np
from scipy.stats import pearsonr

from ...log import get_module_logger, TimeInspector
from .handler import BaseDataHandler
from .launcher import CONFIG_MANAGER
from .fetcher import create_fetcher_with_config
from ...utils import drop_nan_by_y_index, transform_end_date


class BaseTrainer(object):
    def __init__(self, model_class, model_save_path, model_args, data_handler: BaseDataHandler, sacred_ex, **kwargs):
        # 1. Model.
        self.model_class = model_class
        self.model_save_path = model_save_path
        self.model_args = model_args

        # 2. Data handler.
        self.data_handler = data_handler

        # 3. Sacred ex.
        self.ex = sacred_ex

        # 4. Logger.
        self.logger = get_module_logger("Trainer")

        # 5. Data time
        self.train_start_date = kwargs.get("train_start_date", None)
        self.train_end_date = kwargs.get("train_end_date", None)
        self.validate_start_date = kwargs.get("validate_start_date", None)
        self.validate_end_date = kwargs.get("validate_end_date", None)
        self.test_start_date = kwargs.get("test_start_date", None)
        self.test_end_date = transform_end_date(kwargs.get("test_end_date", None))

    @abstractmethod
    def train(self):
        """
        Implement this method indicating how to train a model.
        """
        pass

    @abstractmethod
    def load(self):
        """
        Implement this method indicating how to restore a model and the data.
        """
        pass

    @abstractmethod
    def get_test_pred(self):
        """
        Implement this method indicating how to get prediction result(s) from a model.
        """
        pass

    def get_test_performance(self):
        """
        Implement this method indicating how to get the performance of the model.
        """
        raise NotImplementedError(f"Please implement `get_test_performance`")

    def get_test_score(self):
        """
        Override this method to transfer the predict result(s) into the score of the stock.
        Note: If this is a multi-label training, you need to transfer predict labels into one score.
              Or you can just use the result of `get_test_pred()` (you can also process the result) if this is one label training.
              We use the first column of the result of `get_test_pred()` as default method (regard it as one label training).
        """
        pred = self.get_test_pred()
        pred_score = pd.DataFrame(index=pred.index)
        pred_score["score"] = pred.iloc(axis=1)[0]
        return pred_score


class StaticTrainer(BaseTrainer):
    def __init__(self, model_class, model_save_path, model_args, data_handler, sacred_ex, **kwargs):
        super(StaticTrainer, self).__init__(model_class, model_save_path, model_args, data_handler, sacred_ex, **kwargs)
        self.model = None

        split_data = self.data_handler.get_split_data(
            self.train_start_date,
            self.train_end_date,
            self.validate_start_date,
            self.validate_end_date,
            self.test_start_date,
            self.test_end_date,
        )
        (
            self.x_train,
            self.y_train,
            self.x_validate,
            self.y_validate,
            self.x_test,
            self.y_test,
        ) = split_data

    def train(self):
        TimeInspector.set_time_mark()
        model = self.model_class(**self.model_args)

        if CONFIG_MANAGER.ex_config.finetune:
            fetcher = create_fetcher_with_config(CONFIG_MANAGER, load_form_loader=True)
            loader_model = fetcher.get_experiment(
                exp_name=CONFIG_MANAGER.ex_config.loader_name,
                exp_id=CONFIG_MANAGER.ex_config.loader_id,
                fields=["model"],
            )["model"]

            if isinstance(loader_model, list):
                model_index = (
                    -1
                    if CONFIG_MANAGER.ex_config.loader_model_index is None
                    else CONFIG_MANAGER.ex_config.loader_model_index
                )
                loader_model = loader_model[model_index]

            model.load(loader_model)
            model.finetune(self.x_train, self.y_train, self.x_validate, self.y_validate)
        else:
            model.fit(self.x_train, self.y_train, self.x_validate, self.y_validate)
        model.save(self.model_save_path)
        self.ex.add_artifact(self.model_save_path)
        self.model = model
        TimeInspector.log_cost_time("Finished training model.")

    def load(self):
        model = self.model_class(**self.model_args)

        # Load model
        fetcher = create_fetcher_with_config(CONFIG_MANAGER, load_form_loader=True)
        loader_model = fetcher.get_experiment(
            exp_name=CONFIG_MANAGER.ex_config.loader_name,
            exp_id=CONFIG_MANAGER.ex_config.loader_id,
            fields=["model"],
        )["model"]

        if isinstance(loader_model, list):
            model_index = (
                -1
                if CONFIG_MANAGER.ex_config.loader_model_index is None
                else CONFIG_MANAGER.ex_config.loader_model_index
            )
            loader_model = loader_model[model_index]

        model.load(loader_model)

        # Save model, after load, if you don't save the model, the result of this experiment will be no model
        model.save(self.model_save_path)
        self.ex.add_artifact(self.model_save_path)
        self.model = model

    def get_test_pred(self):
        pred = self.model.predict(self.x_test)
        pred = pd.DataFrame(pred, index=self.x_test.index, columns=self.y_test.columns)
        return pred

    def get_test_performance(self):
        try:
            model_score = self.model.score(self.x_test, self.y_test)
        except NotImplementedError:
            model_score = None
        # Remove rows from x, y and w, which contain Nan in any columns in y_test.
        x_test, y_test, __ = drop_nan_by_y_index(self.x_test, self.y_test)
        pred_test = self.model.predict(x_test)
        model_pearsonr = pearsonr(np.ravel(pred_test), np.ravel(y_test.values))[0]

        performance = {"model_score": model_score, "model_pearsonr": model_pearsonr}
        return performance


class RollingTrainer(BaseTrainer):
    def __init__(self, model_class, model_save_path, model_args, data_handler, sacred_ex, **kwargs):
        super(RollingTrainer, self).__init__(
            model_class, model_save_path, model_args, data_handler, sacred_ex, **kwargs
        )
        self.rolling_period = kwargs.get("rolling_period", 60)
        self.models = []
        self.rolling_data = []
        self.all_x_test = []
        self.all_y_test = []
        for data in self.data_handler.get_rolling_data(
            self.train_start_date,
            self.train_end_date,
            self.validate_start_date,
            self.validate_end_date,
            self.test_start_date,
            self.test_end_date,
            self.rolling_period,
        ):
            self.rolling_data.append(data)
            __, __, __, __, x_test, y_test = data
            self.all_x_test.append(x_test)
            self.all_y_test.append(y_test)

    def train(self):
        # 1. Get total data parts.
        # total_data_parts = self.data_handler.total_data_parts
        # self.logger.warning('Total numbers of model are: {}, start training models...'.format(total_data_parts))
        if CONFIG_MANAGER.ex_config.finetune:
            fetcher = create_fetcher_with_config(CONFIG_MANAGER, load_form_loader=True)
            loader_model = fetcher.get_experiment(
                exp_name=CONFIG_MANAGER.ex_config.loader_name,
                exp_id=CONFIG_MANAGER.ex_config.loader_id,
                fields=["model"],
            )["model"]
            loader_model_index = CONFIG_MANAGER.ex_config.loader_model_index
        previous_model_path = ""
        # 2. Rolling train.
        for (
            index,
            (x_train, y_train, x_validate, y_validate, x_test, y_test),
        ) in enumerate(self.rolling_data):
            TimeInspector.set_time_mark()
            model = self.model_class(**self.model_args)

            if CONFIG_MANAGER.ex_config.finetune:
                # Finetune model
                if loader_model_index is None and isinstance(loader_model, list):
                    try:
                        model.load(loader_model[index])
                    except IndexError:
                        # Load model by previous_model_path
                        with open(previous_model_path, "rb") as fp:
                            model.load(fp)
                        model.finetune(x_train, y_train, x_validate, y_validate)
                else:

                    if index == 0:
                        loader_model = (
                            loader_model[loader_model_index] if isinstance(loader_model, list) else loader_model
                        )
                        model.load(loader_model)
                    else:
                        with open(previous_model_path, "rb") as fp:
                            model.load(fp)

                    model.finetune(x_train, y_train, x_validate, y_validate)

            else:
                model.fit(x_train, y_train, x_validate, y_validate)

            model_save_path = "{}_{}".format(self.model_save_path, index)
            model.save(model_save_path)
            previous_model_path = model_save_path
            self.ex.add_artifact(model_save_path)
            self.models.append(model)
            TimeInspector.log_cost_time("Finished training model: {}.".format(index + 1))

    def load(self):
        """
        Load the data and the model
        """
        fetcher = create_fetcher_with_config(CONFIG_MANAGER, load_form_loader=True)
        loader_model = fetcher.get_experiment(
            exp_name=CONFIG_MANAGER.ex_config.loader_name,
            exp_id=CONFIG_MANAGER.ex_config.loader_id,
            fields=["model"],
        )["model"]
        for index in range(len(self.all_x_test)):
            model = self.model_class(**self.model_args)

            model.load(loader_model[index])

            # Save model
            model_save_path = "{}_{}".format(self.model_save_path, index)
            model.save(model_save_path)
            self.ex.add_artifact(model_save_path)

            self.models.append(model)

    def get_test_pred(self):
        """
        Predict the score on test data with the models.
        Please ensure the models and data are loaded before call this score.

        :return: the predicted scores for the pred
        """
        pred_df_list = []
        y_test_columns = self.all_y_test[0].columns
        # Start iteration.
        for model, x_test in zip(self.models, self.all_x_test):
            pred = model.predict(x_test)
            pred_df = pd.DataFrame(pred, index=x_test.index, columns=y_test_columns)
            pred_df_list.append(pred_df)
        return pd.concat(pred_df_list)

    def get_test_performance(self):
        """
        Get the performances of the models

        :return: the performances of models
        """
        pred_test_list = []
        y_test_list = []
        scorer = self.models[0]._scorer
        for model, x_test, y_test in zip(self.models, self.all_x_test, self.all_y_test):
            # Remove rows from x, y and w, which contain Nan in any columns in y_test.
            x_test, y_test, __ = drop_nan_by_y_index(x_test, y_test)
            pred_test_list.append(model.predict(x_test))
            y_test_list.append(np.squeeze(y_test.values))

        pred_test_array = np.concatenate(pred_test_list, axis=0)
        y_test_array = np.concatenate(y_test_list, axis=0)

        model_score = scorer(y_test_array, pred_test_array)
        model_pearsonr = pearsonr(np.ravel(y_test_array), np.ravel(pred_test_array))[0]

        performance = {"model_score": model_score, "model_pearsonr": model_pearsonr}
        return performance
