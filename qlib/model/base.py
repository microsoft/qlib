# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import abc
from ..utils.serial import Serializable


class BaseModel(Serializable, metaclass=abc.ABCMeta):
    '''Modeling things'''

    @abc.abstractmethod
    def predict(self, *args, **kwargs) -> object:
        """ Make predictions after modeling things """
        pass

    def __call__(self, *args, **kwargs) -> object:
        """ levarge Python syntactic sugar to make the models' behaviors like functions """
        return self.predict(*args, **kwargs)


class Model(BaseModel):
    '''Learnable Models'''

    # TODO: Make the model easier.
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

    @abc.abstractmethod
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
