# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import pandas as pd
from typing import Dict, Iterable


def align_index(df_dict, join):
    res = {}
    for k, df in df_dict.items():
        if join is not None and k != join:
            df = df.reindex(df_dict[join].index)
        res[k] = df
    return res


# Mocking the pd.DataFrame class
class SepDataFrame:
    """
    (Sep)erate DataFrame
    We usually concat multiple dataframe to be processed together(Such as feature, label, weight, filter).
    However, they are usually be used separately at last.
    This will result in extra cost for concatenating and splitting data(reshaping and copying data in the memory is very expensive)

    SepDataFrame tries to act like a DataFrame whose column with multiindex
    """

    def __init__(self, df_dict: Dict[str, pd.DataFrame], join: str, skip_align=False):
        """
        initialize the data based on the dataframe dictionary

        Parameters
        ----------
        df_dict : Dict[str, pd.DataFrame]
            dataframe dictionary
        join : str
            how to join the data
            It will reindex the dataframe based on the join key.
            If join is None, the reindex step will be skipped

        skip_align :
            for some cases, we can improve performance by skipping aligning index
        """
        self.join = join

        if skip_align:
            self._df_dict = df_dict
        else:
            self._df_dict = align_index(df_dict, join)

    @property
    def loc(self):
        return SDFLoc(self, join=self.join)

    @property
    def index(self):
        return self._df_dict[self.join].index

    def apply_each(self, method: str, skip_align=True, *args, **kwargs):
        """
        Assumptions:
        - inplace methods will return None
        """
        inplace = False
        df_dict = {}
        for k, df in self._df_dict.items():
            df_dict[k] = getattr(df, method)(*args, **kwargs)
            if df_dict[k] is None:
                inplace = True
        if not inplace:
            return SepDataFrame(df_dict=df_dict, join=self.join, skip_align=skip_align)

    def sort_index(self, *args, **kwargs):
        return self.apply_each("sort_index", True, *args, **kwargs)

    def copy(self, *args, **kwargs):
        return self.apply_each("copy", True, *args, **kwargs)

    def _update_join(self):
        if self.join not in self:
            self.join = next(iter(self._df_dict.keys()))

    def __getitem__(self, item):
        return self._df_dict[item]

    def __setitem__(self, item: str, df: pd.DataFrame):
        # TODO: consider the join behavior
        self._df_dict[item] = df

    def __delitem__(self, item: str):
        del self._df_dict[item]
        self._update_join()

    def __contains__(self, item):
        return item in self._df_dict

    def __len__(self):
        return len(self._df_dict[self.join])

    def droplevel(self, *args, **kwargs):
        raise NotImplementedError(f"Please implement the `droplevel` method")

    @property
    def columns(self):
        dfs = []
        for k, df in self._df_dict.items():
            df = df.head(0)
            df.columns = pd.MultiIndex.from_product([[k], df.columns])
            dfs.append(df)
        return pd.concat(dfs, axis=1).columns

    # Useless methods
    @staticmethod
    def merge(df_dict: Dict[str, pd.DataFrame], join: str):
        all_df = df_dict[join]
        for k, df in df_dict.items():
            if k != join:
                all_df = all_df.join(df)
        return all_df


class SDFLoc:
    """Mock Class"""

    def __init__(self, sdf: SepDataFrame, join):
        self._sdf = sdf
        self.axis = None
        self.join = join

    def __call__(self, axis):
        self.axis = axis
        return self

    def __getitem__(self, args):
        if self.axis == 1:
            if isinstance(args, str):
                return self._sdf[args]
            elif isinstance(args, (tuple, list)):
                new_df_dict = {k: self._sdf[k] for k in args}
                return SepDataFrame(new_df_dict, join=self.join if self.join in args else args[0], skip_align=True)
            else:
                raise NotImplementedError(f"This type of input is not supported")
        elif self.axis == 0:
            return SepDataFrame(
                {k: df.loc(axis=0)[args] for k, df in self._sdf._df_dict.items()}, join=self.join, skip_align=True
            )
        else:
            df = self._sdf
            if isinstance(args, tuple):
                ax0, *ax1 = args
                if len(ax1) == 0:
                    ax1 = None
                if ax1 is not None:
                    df = df.loc(axis=1)[ax1]
                if ax0 is not None:
                    df = df.loc(axis=0)[ax0]
                return df
            else:
                return df.loc(axis=0)[args]


# Patch pandas DataFrame
# Tricking isinstance to accept SepDataFrame as its subclass
import builtins


def _isinstance(instance, cls):
    if isinstance_orig(instance, SepDataFrame):  # pylint: disable=E0602
        if isinstance(cls, Iterable):
            for c in cls:
                if c is pd.DataFrame:
                    return True
        elif cls is pd.DataFrame:
            return True
    return isinstance_orig(instance, cls)  # pylint: disable=E0602


builtins.isinstance_orig = builtins.isinstance
builtins.isinstance = _isinstance

if __name__ == "__main__":
    sdf = SepDataFrame({}, join=None)
    print(isinstance(sdf, (pd.DataFrame,)))
    print(isinstance(sdf, pd.DataFrame))
