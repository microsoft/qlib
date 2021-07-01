# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import abc
from typing import Union, List, Tuple
from .dataset import MetaDataset


class MetaModel(metaclass=abc.ABCMeta):
    """
    The meta-model controls the training process.
    """

    @abc.abstractmethod
    def fit(self, *args, **kwargs):
        """
        The training process of the meta-model.
        """
        pass

    @abc.abstractmethod
    def inference(self, *args, **kwargs):
        """
        The inference process of the meta-model.
        """
        pass


class MetaTaskModel(MetaModel):
    """
    This type of meta-model deals with base task definitions. The meta-model creates tasks for training new base forecasting models after it is trained. `prepare_tasks` directly modifies the task definitions.
    """

    @abc.abstractmethod
    def prepare_tasks(self, tasks: List[dict]):
        """
        The meta-model modifies the tasks. The function will return the modified task list.

        Parameters
        ----------
        tasks: List[dict]
            A List of task definitions for the meta-model to modify.
        """
        pass


class MetaGuideModel(MetaModel):
    """
    This type of meta-model aims to guide the training process of the base model. The meta-model interacts with the base forecasting models during their training process.
    """

    @abc.abstractmethod
    def fit(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def inference(self, *args, **kwargs):
        pass
