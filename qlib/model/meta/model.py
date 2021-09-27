# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import abc
from typing import Union, List, Tuple

from qlib.model.meta.task import MetaTask
from .dataset import MetaDataset


class MetaModel(metaclass=abc.ABCMeta):
    """
    The meta-model guiding the model learning.

    The word `Guiding` can be categorized into two types based on the stage of model learning
    - The definition of learning tasks:  Please refer to docs of `MetaTaskModel`
    - Controlling the learning process of models: Please refer to the docs of `MetaGuideModel`
    """

    @abc.abstractmethod
    def fit(self, *args, **kwargs):
        """
        The training process of the meta-model.
        """
        pass

    @abc.abstractmethod
    def inference(self, *args, **kwargs) -> object:
        """
        The inference process of the meta-model.

        Returns
        -------
        object:
            Some information to guide the model learning
        """
        pass


class MetaTaskModel(MetaModel):
    """
    This type of meta-model deals with base task definitions. The meta-model creates tasks for training new base forecasting models after it is trained. `prepare_tasks` directly modifies the task definitions.
    """

    @abc.abstractmethod
    def prepare_task(self, task: MetaTask) -> dict:
        """
        Input a meta task and output a task with qlib format

        When modifying the model tasks, the meta model will leverage `self.inference` to get some necessary
        information.

        Parameters
        ----------
        task : MetaTask
            meta task to inference

        Returns
        -------
        dict:
            A task with Qlib format
        """

    # NOTE: factor;   Please justify the necessity of this method
    # @abc.abstractmethod
    # def prepare_tasks(self, tasks: List[dict]) -> List[dict]:
    #     """
    #     The meta-model modifies the tasks. The function will return the modified task list.
    #
    #     Parameters
    #     ----------
    #     tasks: List[dict]
    #         A list of task definitions for the meta-model to modify.
    #
    #     Returns
    #     -------
    #     List[dict]:
    #         A list of modified task definitions.
    #     """
    #     pass


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
