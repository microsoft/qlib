# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
This module covers some utility functions that operate on data or basic object
"""
from copy import deepcopy
from typing import List, Union

import numpy as np
import pandas as pd

from qlib.data.data import DatasetProvider



def robust_zscore(x: pd.Series, zscore=False):
    """Robust ZScore Normalization

    Use robust statistics for Z-Score normalization:
        mean(x) = median(x)
        std(x) = MAD(x) * 1.4826

    Reference:
        https://en.wikipedia.org/wiki/Median_absolute_deviation.
    """
    x = x - x.median()
    mad = x.abs().median()
    x = np.clip(x / mad / 1.4826, -3, 3)
    if zscore:
        x -= x.mean()
        x /= x.std()
    return x


def zscore(x: Union[pd.Series, pd.DataFrame]):
    return (x - x.mean()).div(x.std())


def deepcopy_basic_type(obj: object) -> object:
    """
    deepcopy an object without copy the complicated objects.
        This is useful when you want to generate Qlib tasks and share the handler

    NOTE:
    - This function can't handle recursive objects!!!!!

    Parameters
    ----------
    obj : object
        the object to be copied

    Returns
    -------
    object:
        The copied object
    """
    if isinstance(obj, tuple):
        return tuple(deepcopy_basic_type(i) for i in obj)
    elif isinstance(obj, list):
        return list(deepcopy_basic_type(i) for i in obj)
    elif isinstance(obj, dict):
        return {k: deepcopy_basic_type(v) for k, v in obj.items()}
    else:
        return obj


S_DROP = "__DROP__"  # this is a symbol which indicates drop the value


def update_config(base_config: dict, ext_config: Union[dict, List[dict]]):
    """
    supporting adding base config based on the ext_config

    >>> bc = {"a": "xixi"}
    >>> ec = {"b": "haha"}
    >>> new_bc = update_config(bc, ec)
    >>> print(new_bc)
    {'a': 'xixi', 'b': 'haha'}
    >>> print(bc)  # base config should not be changed
    {'a': 'xixi'}
    >>> print(update_config(bc, {"b": S_DROP}))
    {'a': 'xixi'}
    >>> print(update_config(new_bc, {"b": S_DROP}))
    {'a': 'xixi'}
    """

    base_config = deepcopy(base_config)  # in case of modifying base config

    for ec in ext_config if isinstance(ext_config, (list, tuple)) else [ext_config]:
        for key in ec:
            if key not in base_config:
                # if it is not in the default key, then replace it.
                # ADD if not drop
                if ec[key] != S_DROP:
                    base_config[key] = ec[key]

            else:
                if isinstance(base_config[key], dict) and isinstance(ec[key], dict):
                    # Recursive
                    # Both of them are dict, then update it nested
                    base_config[key] = update_config(base_config[key], ec[key])
                elif ec[key] == S_DROP:
                    # DROP
                    del base_config[key]
                else:
                    # REPLACE
                    # one of then are not dict. Then replace
                    base_config[key] = ec[key]
    return base_config


def guess_horizon(label: List):
    """
    Try to guess the horizon by parsing label
    """
    expr = DatasetProvider.parse_fields(label)[0]
    lft_etd, rght_etd = expr.get_extended_window_size()
    return rght_etd


class DataRange:
    """存储数据范围信息的类，包括股票、时间范围和特征
    
    Attributes
    ----------
    instruments : List[str]
        股票代码列表
    start_time : str
        开始时间
    end_time : str
        结束时间
    fields : List[str]
        特征列表
    """
    
    def __init__(self, instruments: List[str], start_time: str, end_time: str, fields: List[str]):
        self.instruments = instruments
        self.start_time = start_time
        self.end_time = end_time
        self.fields = fields
        
    def __repr__(self):
        return f"DataRange(instruments={self.instruments}, start_time={self.start_time}, end_time={self.end_time}, fields={self.fields})"
    
    @classmethod
    def from_expression(cls, expression: str) -> 'DataRange':
        """从表达式创建 DataRange 对象
        
        Parameters
        ----------
        expression : str
            表达式字符串，例如 "$close" 或 "$open+$close"
            
        Returns
        -------
        DataRange
            包含数据范围信息的对象
        """
        # 解析表达式获取特征
        fields = DatasetProvider.parse_fields([expression])
        
        # 获取时间范围
        left_etd, right_etd = fields[0].get_extended_window_size()
        
        # 这里需要根据实际情况设置股票列表和时间范围
        # 这里只是示例，实际使用时需要根据具体需求修改
        instruments = []  # 需要从配置或其他地方获取
        start_time = ""   # 需要从配置或其他地方获取
        end_time = ""     # 需要从配置或其他地方获取
        
        return cls(instruments, start_time, end_time, fields)
