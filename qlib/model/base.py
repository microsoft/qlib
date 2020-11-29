# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import abc
from ..utils.serial import Serializable
from ..data.dataset import Dataset


class BaseModel(Serializable, metaclass=abc.ABCMeta):
    """Modeling things"""

    @abc.abstractmethod
    def predict(self, *args, **kwargs) -> object:
        """ Make predictions after modeling things """
        pass

    def __call__(self, *args, **kwargs) -> object:
        """ leverage Python syntactic sugar to make the models' behaviors like functions """
        return self.predict(*args, **kwargs)


class Model(BaseModel):
    """Learnable Models"""

    def fit(self, dataset: Dataset):
        """
        Learn model from the base model

        .. note::

            The the attribute names of learned model should `not` start with '_'. So that the model could be
            dumped to disk.

        Parameters
        ----------
        dataset : Dataset
            dataset will generate the processed data from model training.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def predict(self, dataset: Dataset) -> object:
        """give prediction given Dataset

        Parameters
        ----------
        dataset : Dataset
            dataset will generate the processed dataset from model training.
        """
        raise NotImplementedError()


class ModelFT(Model):
    """Model (F)ine(t)unable"""

    @abc.abstractmethod
    def finetune(self, dataset: Dataset):
        """finetune model based given dataset

        A typical use case of finetuning model with qlib.workflow.R

        .. code-block:: python

            # start exp to train init model
            with R.start(experiment_name="init models"):
                model.fit(dataset)
                R.save_objects(init_model=model)
                rid = R.get_recorder().id

            # Finetune model based on previous trained model
            with R.start(experiment_name="finetune model"):
                recorder = R.get_recorder(rid, experiment_name="init models")
                model = recorder.load_object("init_model")
                model.finetune(dataset, num_boost_round=10)


        Parameters
        ----------
        dataset : Dataset
            dataset will generate the processed dataset from model training.
        """
        raise NotImplementedError()
