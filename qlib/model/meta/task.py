# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import abc
from typing import Union, List, Tuple
from ...data.dataset import DatasetH, TSDatasetH
from ...utils import init_instance_by_config


class MetaTask(metaclass=abc.ABCMeta):
    """
    A single meta-task, a meta-dataset contains a list of them.
    """

    def __init__(self, dataset_dict: dict, *args, **kwargs):
        """

        Parameters
        ----------
        dataset_dict: dict
            The dataset definition for this meta-task instance.
        """
        self.dataset_dict = dataset_dict
        self.dataset = init_instance_by_config(self.dataset_dict)

    def get_dataset(self) -> Union[DatasetH, TSDatasetH]:
        """
        Get the dataset instance defined in the meta-task.

        Returns
        -------
        Union[DatasetH, TSDatasetH]:
            The instance of the dataset definition.
        """
        return self.dataset

    @abc.abstractmethod
    def prepare_task_data(self):
        """
        Prepare the data for training the meta-model.
        """
        pass
