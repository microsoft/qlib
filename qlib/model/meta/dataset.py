# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import abc
from typing import Union, List, Tuple, Text
from ...workflow.task.gen import RollingGen, task_generator
from ...data.dataset.handler import DataHandler
from ...utils.serial import Serializable


class MetaDataset(Serializable, metaclass=abc.ABCMeta):
    """
    A dataset fetching the data in a meta-level.
    """

    def __init__(self, *args, **kwargs):
        """
        The meta-dataset maintains a list of meta-tasks when it is initialized.
        """
        super().__init__(*args, **kwargs)
        self.meta_tasks = []

    @abc.abstractmethod
    def prepare_tasks(self, segments: Union[List[Text], Tuple[Text], Text], *args, **kwargs):
        """
        Prepare the data in each meta-task and ready for training.

        The following code example shows how to retrieve a list of meta-tasks from the `meta_dataset`:

            .. code-block:: Python

                # get the train segment and the test segment, both of them are lists
                train_meta_tasks, test_meta_tasks = meta_dataset.prepare_tasks(["train", "test"])

        Returns
        -------
        list:
            A list of the prepared data of each meta-task for training the meta-model. For multiple segments [seg1, seg2, ... , segN], the returned list will be [[tasks in seg1], [tasks in seg2], ... , [tasks in segN]].
        """
        pass


class MetaDatasetH(MetaDataset):
    """
    MetaDataset with specified DataHandler.
    """

    def __init__(self, data_handler: DataHandler, *args, **kwargs):
        """

        Parameters
        ----------
        data_handler: DataHandler
            The shared DataHandler among meta-tasks.
        """
        super().__init__(*args, **kwargs)
        self.data_handler = data_handler
