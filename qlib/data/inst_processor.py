import abc
import json
import pandas as pd


class InstProcessor:
    @abc.abstractmethod
    def __call__(self, df: pd.DataFrame, *args, **kwargs):
        """
        process the data

        NOTE: **The processor could change the content of `df` inplace !!!!! **
        User should keep a copy of data outside

        Parameters
        ----------
        df : pd.DataFrame
            The raw_df of handler or result from previous processor.
        """
        pass

    def __str__(self):
        return f"{self.__class__.__name__}:{json.dumps(self.__dict__, sort_keys=True, default=str)}"
