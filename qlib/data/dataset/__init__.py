from ...utils.serial import Serializable
from typing import Union, List, Tuple
from ...utils import init_instance_by_config
from ...log import get_module_logger
from .handler import DataHandler, DataHandlerLP
from inspect import getfullargspec
import pandas as pd


class Dataset(Serializable):
    """
    Preparing data for model training and inferencing.
    """

    def __init__(self, *args, **kwargs):
        """
        init is designed to finish following steps:

        - setup data
            - The data related attributes' names should start with '_' so that it will not be saved on disk when serializing.

        - initialize the state of the dataset(info to prepare the data)
            - The name of essential state for preparing data should not start with '_' so that it could be serialized on disk when serializing.

        The data could specify the info to caculate the essential data for preparation
        """
        self.setup_data(*args, **kwargs)
        super().__init__()

    def setup_data(self, *args, **kwargs):
        """
        Setup the data.

        We split the setup_data function for following situation:

        - User have a Dataset object with learned status on disk.

        - User load the Dataset object from the disk(Note the init function is skiped).

        - User call `setup_data` to load new data.

        - User prepare data for model based on previous status.
        """
        pass

    def prepare(self, *args, **kwargs) -> object:
        """
        The type of dataset depends on the model. (It could be pd.DataFrame, pytorch.DataLoader, etc.)
        The parameters should specify the scope for the prepared data
        The method should:
        - process the data

        - return the processed data

        Returns
        -------
        object:
            return the object
        """
        pass


class DatasetH(Dataset):
    """
    Dataset with Data(H)andler

    User should try to put the data preprocessing functions into handler.
    Only following data processing functions should be placed in Dataset:

    - The processing is related to specific model.

    - The processing is related to data split.
    """

    def __init__(self, handler: Union[dict, DataHandler], segments: list):
        """
        Parameters
        ----------
        handler : Union[dict, DataHandler]
            handler will be passed into setup_data.
        segments : list
            handler will be passed into setup_data.
        """
        super().__init__(handler, segments)

    def setup_data(self, handler: Union[dict, DataHandler], segments: list):
        """
        Setup the underlying data.

        Parameters
        ----------
        handler : Union[dict, DataHandler]
            handler could be:

            - insntance of `DataHandler`

            - config of `DataHandler`.  Please refer to `DataHandler`

        segments : list
            Describe the options to segment the data.
            Here are some examples:

            .. code-block::

                1) 'segments': {
                        'train': ("2008-01-01", "2014-12-31"),
                        'valid': ("2017-01-01", "2020-08-01",),
                        'test': ("2015-01-01", "2016-12-31",),
                    }
                2) 'segments': {
                        'insample': ("2008-01-01", "2014-12-31"),
                        'outsample': ("2017-01-01", "2020-08-01",),
                    }
        """
        self._handler = init_instance_by_config(handler, accept_types=DataHandler)
        self._segments = segments.copy()

    def prepare(
        self,
        segments: Union[List[str], Tuple[str], str, slice],
        col_set=DataHandler.CS_ALL,
        data_key=DataHandlerLP.DK_I,
        **kwargs,
    ) -> Union[List[pd.DataFrame], pd.DataFrame]:
        """
        Prepare the data for learning and inference.

        Parameters
        ----------
        segments : Union[List[str], Tuple[str], str, slice]
            Describe the scope of the data to be prepared
            Here are some examples:

            - 'train'

            - ['train', 'valid']

        col_set : str
            The col_set will be passed to self._handler when fetching data.
        data_key : str
            The data to fetch:  DK_*
            Default is DK_I, which indicate fetching data for **inference**.

        Returns
        -------
        Union[List[pd.DataFrame], pd.DataFrame]:

        Raises
        ------
        NotImplementedError:
        """
        logger = get_module_logger("DatasetH")
        fetch_kwargs = {"col_set": col_set}
        fetch_kwargs.update(kwargs)
        if "data_key" in getfullargspec(self._handler.fetch).args:
            fetch_kwargs["data_key"] = data_key
        else:
            logger.info(f"data_key[{data_key}] is ignored.")

        if isinstance(segments, (list, tuple)):
            return [self._handler.fetch(slice(*self._segments[seg]), **fetch_kwargs) for seg in segments]
        elif isinstance(segments, str):
            return self._handler.fetch(slice(*self._segments[segments]), **fetch_kwargs)
        else:
            raise NotImplementedError(f"This type of input is not supported")
