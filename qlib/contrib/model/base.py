# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import division
from __future__ import print_function

import abc
import six


@six.add_metaclass(abc.ABCMeta)
class Model(object):
    """Model base class"""

    @property
    def name(self):
        return type(self).__name__

    def fit(self, x_train, y_train, x_valid, y_valid, w_train=None, w_valid=None, **kwargs):
        """fix train with cross-validation
        Fit model when ex_config.finetune is False

        Parameters
        ----------
        x_train : pd.dataframe
            train data
        y_train : pd.dataframe
            train label
        x_valid : pd.dataframe
            valid data
        y_valid : pd.dataframe
            valid label
        w_train : pd.dataframe
            train weight
        w_valid : pd.dataframe
            valid weight

        Returns
        ----------
        Model
            trained model
        """
        raise NotImplementedError()

    def score(self, x_test, y_test, w_test=None, **kwargs):
        """evaluate model with test data/label

        Parameters
        ----------
        x_test : pd.dataframe
            test data
        y_test : pd.dataframe
            test label
        w_test : pd.dataframe
            test weight

        Returns
        ----------
        float
            evaluation score
        """
        raise NotImplementedError()

    def predict(self, x_test, **kwargs):
        """predict given test data

        Parameters
        ----------
        x_test : pd.dataframe
            test data

        Returns
        ----------
        np.ndarray
            test predict label
        """
        raise NotImplementedError()

    def save(self, fname, **kwargs):
        """save model

        Parameters
        ----------
        fname : str
            model filename
        """
        # TODO: Currently need to save the model as a single file, otherwise the estimator may not be compatible
        raise NotImplementedError()

    def load(self, buffer, **kwargs):
        """load model

        Parameters
        ----------
        buffer : bytes
            binary data of model parameters

        Returns
        ----------
        Model
            loaded model
        """
        raise NotImplementedError()

    def get_data_with_date(self, date, **kwargs):
        """
        Will be called in online module
        need to return the data that used to predict the label (score) of stocks at date.

        :param
            date: pd.Timestamp
                predict date
        :return:
            data: the input data that used to predict the label (score) of stocks at predict date.
        """
        raise NotImplementedError("get_data_with_date for this model is not implemented.")

    def finetune(self, x_train, y_train, x_valid, y_valid, w_train=None, w_valid=None, **kwargs):
        """Finetune model
        In `RollingTrainer`:
            if loader.model_index is None:
                If provide 'Static Model', based on the provided 'Static' model update.
                If provide 'Rolling Model', skip the model of load, based on the last 'provided model' update.

            if loader.model_index is not None:
                Based on the provided model(loader.model_index) update.

        In `StaticTrainer`:
            If the load is 'static model':
                Based on the 'static model' update
            If the load is 'rolling model':
                Based on the provided model(`loader.model_index`) update. If `loader.model_index` is None, use the last model.

        Parameters
        ----------
        x_train : pd.dataframe
            train data
        y_train : pd.dataframe
            train label
        x_valid : pd.dataframe
            valid data
        y_valid : pd.dataframe
            valid label
        w_train : pd.dataframe
            train weight
        w_valid : pd.dataframe
            valid weight

        Returns
        ----------
        Model
            finetune model
        """
        raise NotImplementedError("Finetune for this model is not implemented.")
