# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import abc
from typing import Dict, Union, List, Tuple, Text
from ...workflow.task.gen import RollingGen, task_generator
from ...data.dataset.handler import DataHandler
from ...utils.serial import Serializable


class MetaDataset(Serializable, metaclass=abc.ABCMeta):
    """
    A dataset fetching the data in a meta-level.

    A Meta Dataset is responsible for
    - input a specific task and prepare input data (based a given task) for meta model
    - prepare underlayer data:

    The learnt pattern could transfer to other meta dataset. The following cases should be supported
    - A meta-model trained on meta-dataset A and then applied to meta-dataset B
        - Some pattern are shared between meta-dataset A and B, so meta-input on meta-dataset A are used when meta model are applied on meta-dataset-B
    """

    def __init__(self, segments: Union[Dict[Text, Tuple], float], *args, **kwargs):
        """
        The meta-dataset maintains a list of meta-tasks when it is initialized.

        The segments indicates the way to divide the data

        The duty of the `__init__` function of MetaDataset
        - initialize the tasks
        """
        super().__init__(*args, **kwargs)
        self.segments = segments

    def prepare_tasks(self, segments: Union[List[Text], Text], *args, **kwargs) -> List:
        """
        Prepare the data in each meta-task and ready for training.

        The following code example shows how to retrieve a list of meta-tasks from the `meta_dataset`:

            .. code-block:: Python

                # get the train segment and the test segment, both of them are lists
                train_meta_tasks, test_meta_tasks = meta_dataset.prepare_tasks(["train", "test"])

        Parameters
        ----------
        segments: Union[List[Text], Tuple[Text], Text]
            the info to select data

        Returns
        -------
        list:
            A list of the prepared data of each meta-task for training the meta-model. For multiple segments [seg1, seg2, ... , segN], the returned list will be [[tasks in seg1], [tasks in seg2], ... , [tasks in segN]].
            Each task is a meta task
        """
        if isinstance(segments, (list, tuple)):
            return [self._prepare_seg(seg) for seg in segments]
        elif isinstance(segments, str):
            return self._prepare_seg(segments)
        else:
            raise NotImplementedError(f"This type of input is not supported")

    @abc.abstractmethod
    def _prepare_seg(self, segment: Text):
        """
        prepare a single segment of data for training data

        Parameters
        ----------
        seg : Text
            the name of the segment
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
