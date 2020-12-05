from ...utils.serial import Serializable
from typing import Union, List, Tuple
from ...utils import init_instance_by_config, np_ffill
from ...log import get_module_logger
from .handler import DataHandler, DataHandlerLP
from inspect import getfullargspec
import pandas as pd
import numpy as np
import bisect
from ...utils import lazy_sort_index
from .utils import get_level_index


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

    def _prepare_seg(self, slc: slice, **kwargs):
        """
        Give a slice, retrieve the according data

        Parameters
        ----------
        slc : slice
        """
        return self._handler.fetch(slc, **kwargs)

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

        # Handle all kinds of segments format
        if isinstance(segments, (list, tuple)):
            return [self._prepare_seg(slice(*self._segments[seg]), **fetch_kwargs) for seg in segments]
        elif isinstance(segments, str):
            return self._prepare_seg(slice(*self._segments[segments]), **fetch_kwargs)
        elif isinstance(segments, slice):
            return self._prepare_seg(segments, **fetch_kwargs)
        else:
            raise NotImplementedError(f"This type of input is not supported")


class TSDataSampler:
    """
    (T)ime-(S)eries DataSampler
    This is the result of TSDatasetH

    It works like `torch.data.utils.Dataset`, it provides a very convient interface for constructing time-series
    dataset based on tabular data.

    If user have further requirements for processing data, user could process

    """

    def __init__(self, data: pd.DataFrame, start, end, step_len: int, fillna_type: str = "none"):
        """
        Build a dataset which looks like torch.data.utils.Dataset.

        Parameters
        ----------
        data : pd.DataFrame
            The raw tabular data
        start :
            The indexable start time
        end :
            The indexable end time
        step_len : int
            The length of the time-series step
        fillna_type : int
            How will qlib handle the sample if there is on sample in a specific date.
            none:
                fill with np.nan
            ffill:
                ffill with previous sample
            ffill+bfill:
                ffill with previous samples first and fill with later samples second
        """
        self.start = start
        self.end = end
        self.step_len = step_len
        self.fillna_type = fillna_type
        assert get_level_index(data, "datetime") == 0
        self.data = lazy_sort_index(data)
        # The index of usable data is between start_idx and end_idx
        self.start_idx, self.end_idx = self.data.index.slice_locs(start=pd.Timestamp(start), end=pd.Timestamp(end))
        # self.index_link = self.build_link(self.data)
        self.idx_df, self.idx_map = self.build_index(self.data)

    def config(self, **kwargs):
        # Config the attributes
        for k, v in kwargs.items():
            setattr(self, k, v)

    @staticmethod
    def build_index(data: pd.DataFrame) -> dict:
        """
        The relation of the data

        Parameters
        ----------
        data : pd.DataFrame
            The dataframe with <datetime, DataFrame>

        Returns
        -------
        dict:
            {<index>: <prev_index or None>}
            # get the previous index of a line given index
        """
        # object incase of pandas converting int to flaot
        idx_df = pd.Series(range(data.shape[0]), index=data.index, dtype=np.object)
        idx_df = lazy_sort_index(idx_df.unstack())
        # NOTE: the correctness of `__getitem__` depends on columns sorted here
        idx_df = lazy_sort_index(idx_df, axis=1)

        idx_map = {}
        for i, (_, row) in enumerate(idx_df.iterrows()):
            for j, real_idx in enumerate(row):
                if not np.isnan(real_idx):
                    idx_map[real_idx] = (i, j)
        return idx_df, idx_map

    def __getitem__(self, idx: Union[int, Tuple[object, str]]):
        """
        # We have two method to get the time-series of a sample
        tsds is a instance of TSDataSampler

        # 1) sample by int index directly
        tsds[len(tsds) - 1]

        # 2) sample by <datetime,instrument> index
        tsds['2016-12-31', "SZ300315"]

        # The return value will be similar to the data retrieved by following code
        df.loc(axis=0)['2015-01-01':'2016-12-31', "SZ300315"].iloc[-30:]

        Parameters
        ----------
        idx : Union[int, Tuple[object, str]]
        """
        # The the right row number `i` and col number `j` in idx_df
        if isinstance(idx, (int, np.integer)):
            real_idx = self.start_idx + idx
            if self.start_idx <= real_idx < self.end_idx:
                i, j = self.idx_map[real_idx]
            else:
                raise KeyError(f"{real_idx} is out of [{self.start_idx}, {self.end_idx})")
        elif isinstance(idx, tuple):
            # <TSDataSampler object>["datetime", "instruments"]
            date, inst = idx
            date = pd.Timestamp(date)
            i = bisect.bisect_right(self.idx_df.index, date) - 1
            # NOTE: This relies on the idx_df columns sorted in `__init__`
            j = bisect.bisect_left(self.idx_df.columns, inst)
        else:
            raise NotImplementedError(f"This type of input is not supported")

        data_l = []
        indices = self.idx_df.values[max(i - self.step_len + 1, 0) : i + 1, j]
        indices = indices.reshape(-1)

        if len(indices) < self.step_len:
            indices = np.concatenate([np.full((self.step_len - len(indices),), np.nan), indices])

        if self.fillna_type == "ffill":
            indices = np_ffill(indices)
        elif self.fillna_type == "ffill+bfill":
            indices = np_ffill(np_ffill(indices)[::-1])[::-1]
        else:
            assert self.fillna_type == "none"

        if np.isnan(indices.astype(np.float)).sum() == 0:  # np.isnan only works on np.float
            # All the index exists
            return self.data.values[indices.astype(np.int)]
        else:
            # Only part index exists. These days will be filled with nan
            for idx in indices:
                if np.isnan(idx):
                    data_l.append(np.full((self.data.shape[1],), np.nan))
                else:
                    data_l.append(self.data.values[idx])
            return np.array(data_l)

    def __len__(self):
        return self.end_idx - self.start_idx


class TSDatasetH(DatasetH):
    """
    (T)ime-(S)eries Dataset (H)andler


    Covnert the tabular data to Time-Series data

    Requirements analysis

    The typical workflow of a user to get time-series data for an sample
    - process features
    - slice proper data from data handler:  dimension of sample <feature, >
    - Build relation of samples by <time, instrument> index
        - Be able to sample times series of data <timestep, feature>
        - It will be better if the interface is like "torch.utils.data.Dataset"
    - User could build customized batch based on the data
        - The dimension of a batch of data <batch_idx, feature, timestep>
    """

    def __init__(self, step_len=30, *args, **kwargs):
        self.step_len = step_len
        super().__init__(*args, **kwargs)

    def setup_data(self, *args, **kwargs):
        super().setup_data(*args, **kwargs)
        cal = self._handler.fetch(col_set=self._handler.CS_RAW).index.get_level_values("datetime").unique()
        cal = sorted(cal)
        # Get the datatime index for building timestamp
        self.cal = cal

    def _prepare_seg(self, slc: slice, **kwargs) -> TSDataSampler:
        # Dataset decide how to slice data(Get more data for timeseries).
        start, end = slc.start, slc.stop
        start_idx = bisect.bisect_left(self.cal, pd.Timestamp(start))
        pad_start_idx = max(0, start_idx - self.step_len)
        pad_start = self.cal[pad_start_idx]

        # TSDatasetH will retrieve more data for complete
        data = super()._prepare_seg(slice(pad_start, end), **kwargs)

        tsds = TSDataSampler(data=data, start=start, end=end, step_len=self.step_len)
        return tsds
