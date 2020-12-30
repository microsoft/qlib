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

            The attribute names of learned model should `not` start with '_'. So that the model could be
            dumped to disk.

        The following code example shows how to retrieve `x_train`, `y_train` and `w_train` from the `dataset`:

            .. code-block:: Python

                # get features and labels
                df_train, df_valid = dataset.prepare(
                    ["train", "valid"], col_set=["feature", "label"], data_key=DataHandlerLP.DK_L
                )
                x_train, y_train = df_train["feature"], df_train["label"]
                x_valid, y_valid = df_valid["feature"], df_valid["label"]

                # get weights
                try:
                    wdf_train, wdf_valid = dataset.prepare(["train", "valid"], col_set=["weight"], data_key=DataHandlerLP.DK_L)
                    w_train, w_valid = wdf_train["weight"], wdf_valid["weight"]
                except KeyError as e:
                    w_train = pd.DataFrame(np.ones_like(y_train.values), index=y_train.index)
                    w_valid = pd.DataFrame(np.ones_like(y_valid.values), index=y_valid.index)

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

        Returns
        -------
        Prediction results with certain type such as `pandas.Series`.
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
