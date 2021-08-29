import abc
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


class ResampleProcessor(InstProcessor):
    """resample data"""

    def __init__(self, freq: str, func: str, *args, **kwargs):
        self.freq = freq
        self.func = func

    def __call__(self, df: pd.DataFrame, *args, **kwargs):
        return getattr(df.resample(self.freq, level="datetime"), self.func)().dropna(how="all")
