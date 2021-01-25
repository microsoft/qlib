import pandas as pd
import abc
from typing import Union


class Detector(metaclass=abc.ABCMeta):
    """
    Detector is stateful

    The input of detector is  Series with shape <time> or Dataframe with shape <time, factor>
    """

    @abc.abstractmethod
    def check(self, df: Union[pd.Series, pd.DataFrame]) -> pd.Series:
        """
        Check the result of values

        Parameters
        ----------
        df : Union[pd.Series, pd.DataFrame]
            Data to be checked

        Returns
        -------
        pd.Series:
            True: Abnormalities detected.
            False: No abnormality detected.
        """
        pass

    def fit(self, df: pd.DataFrame):
        raise NotImplementedError(f"This type of input is not supported")

    def update(self, df: pd.DataFrame):
        """
        The state of detector can be updated gradually.
        """
        raise NotImplementedError(f"This type of input is not supported")


class NDDetector(Detector):
    """
    Normal Distribution Detector
    This will be used more in offline detector
    """

    def __init__(self, n=3):
        """
        The detection range:
        - mean ± n * std
        """
        self.n = n

    def fit(self, s: pd.Series):
        self.mean = s.mean()
        self.std = s.std()

    def check(self, s: pd.Series):
        return ~s.between(self.mean - self.std * self.n, self.mean + self.std * self.n)


class SWNDD(Detector):
    """(S)liding (W)indow (N)ormal (D)istribition (D)etector"""

    def __init__(self, n=3, **rolling_args):
        self.rolling_args = rolling_args
        self.n = n

    def fit(self, s: pd.Series):
        # TODO: pd.Dataframe is not supported now
        self.mean = s.rolling(**self.rolling_args).mean()
        self.std = s.rolling(**self.rolling_args).std()

    def check(self, s: pd.Series):
        res = ~s.between(self.mean - self.std * self.n, self.mean + self.std * self.n)
        res = res & ~self.mean.isna() & ~self.std.isna()

        return res.reindex(s.index)


# TODO: daily normalization detector.


class CountD(Detector):
    """
    Count detector
    """

    # TODO: check if the number of counts is enough
    # TODO: This is a instance of Count Detector.


class ThresholdD(Detector):
    """
    Threshold (D)etector
    """

    def __init__(self, threshold, reverse=False):
        self.threshold = threshold
        self.reverse = reverse

    def check(self, s: pd.Series):
        return (s < self.threshold) if self.reverse else (s > self.threshold)
