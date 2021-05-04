import pandas as pd
from typing import Union
from abc import abstractmethod, ABCMeta


def format_conv(df: pd.Series, **col_group):
    # TODO: col_group
    df = df.copy()  # performance problems here
    if len(col_group) > 0:
        col_group_df = pd.DataFrame({name: group.reindex(df.index) for name, group in col_group.items()})
        col_group_df = col_group_df.reindex(df.index)
        # merge all the groups into df.index
        col_group_df = col_group_df.set_index(keys=col_group_df.columns.to_list(), append=True)
        df.index = col_group_df.index

    ustk_cols = [col for col in df.index.names if col != "datetime"]
    return df.unstack(ustk_cols)


class MetricExt(metaclass=ABCMeta):
    """Metric Extractor
    Current design.
    The input data are assumed like qlib format
    The extracted information like time-series. Column could be multiple index
    """

    @abstractmethod
    def extract(self, df: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
        pass


# overall metrics
class AggMetrics(MetricExt):
    """
    TODO: this metric assumes that the daily assumptions(The operation is used on each row)
    """

    def __init__(self, group=None):
        if isinstance(group, str):
            group = [group]
        self.group = group

    def extract(self, df: pd.DataFrame) -> pd.Series:
        if self.group is None:
            return df.apply(self.agg, axis=1)
        else:
            return df.groupby(self.group, axis=1).apply(self.agg, axis=1)

    @abstractmethod
    def agg(self, *args, **kwargs):
        pass


class StdM(AggMetrics):
    def agg(self, s, *args, **kwargs):
        return s.std(*args, **kwargs)


class MeanM(AggMetrics):
    def agg(self, s, *args, **kwargs):
        return s.mean(*args, **kwargs)


class SkewM(AggMetrics):
    def agg(self, s, *args, **kwargs):
        return s.skew(*args, **kwargs)


class KurtM(AggMetrics):
    def agg(self, s, *args, **kwargs):
        return pd.DataFrame.kurt(s, *args, **kwargs)


# sliding window metrics
class SWMetrics(MetricExt):
    """
    (S)liding (W)indow Metrics

    TODO: testing this class
    """

    def __init__(self, **rolling_args):
        self.rolling_args = rolling_args

    def extract(self, df: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
        if isinstance(pd.Series):
            return df.rolling(**self.rolling_args).apply(self.agg)
        elif isinstance(pd.DataFrame):
            return df.rolling(**self.rolling_args).apply(self.agg, axis=0)
        else:
            raise NotImplementedError(f"This type of input is not supported")

    @abstractmethod
    def agg(self, *args, **kwargs):
        pass


## TODO: more metrics is ignored: mean, std, skew, kurt


def calc_corr(df1: pd.DataFrame, df2: pd.DataFrame, mode):
    corr = {}
    for (t1, s1), (t2, s2) in zip(df1.iterrows(), df2.iterrows()):
        assert t1 == t2
        corr[t1] = s1.corr(s2, method=mode)
    return pd.Series(corr)


class AutoCM(MetricExt):
    """(A)uto (C)orrelation (M)etrics"""

    def __init__(self, mode="pearson", shift=1):
        self.mode = mode
        self.shift = shift

    def extract(self, df: pd.DataFrame):
        return calc_corr(df, df.shift(self.shift), self.mode)


class CorrM(MetricExt):
    """correlation extractor """

    def __init__(self, mode="pearson"):
        self.mode = mode

    def extract(self, df1: pd.DataFrame, df2: pd.DataFrame):
        return calc_corr(df1, df2, self.mode)
