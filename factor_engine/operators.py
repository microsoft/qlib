"""
一组对 DataContainer 进行操作的纯函数库。
每个函数都接收 DataContainer 和其他参数，并返回一个新的 DataContainer。
"""
import numpy as np
import pandas as pd
from factor_engine.data_layer.containers import PanelContainer

# --- Binary Operators ---

def op_add(a: PanelContainer, b: PanelContainer) -> PanelContainer:
    """对两个 PanelContainer 的数据执行元素级加法。"""
    if not isinstance(a, PanelContainer) or not isinstance(b, PanelContainer):
        raise TypeError("操作数 'a' 和 'b' 都必须是 PanelContainer。")
    result_data = a.get_data().add(b.get_data())
    return PanelContainer(result_data)

def op_sub(a: PanelContainer, b: PanelContainer) -> PanelContainer:
    """对两个 PanelContainer 的数据执行元素级减法。"""
    if not isinstance(a, PanelContainer) or not isinstance(b, PanelContainer):
        raise TypeError("操作数 'a' 和 'b' 都必须是 PanelContainer。")
    result_data = a.get_data().sub(b.get_data())
    return PanelContainer(result_data)

def op_mul(a: PanelContainer, b: PanelContainer) -> PanelContainer:
    """对两个 PanelContainer 的数据执行元素级乘法。"""
    if not isinstance(a, PanelContainer) or not isinstance(b, PanelContainer):
        raise TypeError("操作数 'a' 和 'b' 都必须是 PanelContainer。")
    result_data = a.get_data().mul(b.get_data())
    return PanelContainer(result_data)

def op_div(a: PanelContainer, b: PanelContainer) -> PanelContainer:
    """对两个 PanelContainer 的数据执行元素级除法。"""
    if not isinstance(a, PanelContainer) or not isinstance(b, PanelContainer):
        raise TypeError("操作数 'a' 和 'b' 都必须是 PanelContainer。")
    a_data = a.get_data()
    b_data = b.get_data()
    # a/b, 用 NaN 替换无穷大值，然后用 0 填充所有 NaN
    result_data = a_data.div(b_data).replace([np.inf, -np.inf], np.nan).fillna(0)
    return PanelContainer(result_data)

def op_max(a: PanelContainer, b: PanelContainer) -> PanelContainer:
    """对两个 PanelContainer 的数据执行元素级最大值。"""
    if not isinstance(a, PanelContainer) or not isinstance(b, PanelContainer):
        raise TypeError("操作数 'a' 和 'b' 都必须是 PanelContainer。")
    result_data = np.maximum(a.get_data(), b.get_data())
    return PanelContainer(result_data)

def op_min(a: PanelContainer, b: PanelContainer) -> PanelContainer:
    """对两个 PanelContainer 的数据执行元素级最小值。"""
    if not isinstance(a, PanelContainer) or not isinstance(b, PanelContainer):
        raise TypeError("操作数 'a' 和 'b' 都必须是 PanelContainer。")
    result_data = np.minimum(a.get_data(), b.get_data())
    return PanelContainer(result_data)

# --- Unary Operators ---

def op_neg(data: PanelContainer) -> PanelContainer:
    """对 PanelContainer 的数据执行元素级取负。"""
    if not isinstance(data, PanelContainer):
        raise TypeError("操作数 'data' 必须是 PanelContainer。")
    return PanelContainer(-data.get_data())

def op_abs(data: PanelContainer) -> PanelContainer:
    """对 PanelContainer 的数据执行元素级绝对值。"""
    if not isinstance(data, PanelContainer):
        raise TypeError("操作数 'data' 必须是 PanelContainer。")
    return PanelContainer(data.get_data().abs())

def op_log(data: PanelContainer) -> PanelContainer:
    """对 PanelContainer 的数据执行元素级自然对数。"""
    if not isinstance(data, PanelContainer):
        raise TypeError("操作数 'data' 必须是 PanelContainer。")
    # log of non-positive numbers is undefined, results in NaN
    result_data = np.log(data.get_data())
    return PanelContainer(result_data.replace([np.inf, -np.inf], np.nan))


# --- Time-Series Operators ---

def op_ts_mean(data: PanelContainer, window: int) -> PanelContainer:
    """计算面板数据在时间序列上的滚动平均值。"""
    if not isinstance(data, PanelContainer):
        raise TypeError("操作数 'data' 必须是 PanelContainer。")
    if not isinstance(window, int) or window <= 0:
        raise TypeError("参数 'window' 必须是正整数。")
    result_data = data.get_data().rolling(window=window, min_periods=1).mean()
    return PanelContainer(result_data)

def op_ts_std(data: PanelContainer, window: int) -> PanelContainer:
    """计算面板数据在时间序列上的滚动标准差。"""
    if not isinstance(data, PanelContainer):
        raise TypeError("操作数 'data' 必须是 PanelContainer。")
    if not isinstance(window, int) or window <= 0:
        raise TypeError("参数 'window' 必须是正整数。")
    result_data = data.get_data().rolling(window=window, min_periods=1).std()
    return PanelContainer(result_data)

def op_shift(data: PanelContainer, window: int) -> PanelContainer:
    """将面板数据在时间序列上向前平移（滞后）。"""
    if not isinstance(data, PanelContainer):
        raise TypeError("操作数 'data' 必须是 PanelContainer。")
    if not isinstance(window, int):
        raise TypeError("参数 'window' 必须是整数。")
    result_data = data.get_data().shift(periods=window)
    return PanelContainer(result_data)

def op_ts_max(data: PanelContainer, window: int) -> PanelContainer:
    """计算面板数据在时间序列上的滚动最大值。"""
    if not isinstance(data, PanelContainer):
        raise TypeError("操作数 'data' 必须是 PanelContainer。")
    if not isinstance(window, int) or window <= 0:
        raise TypeError("参数 'window' 必须是正整数。")
    result_data = data.get_data().rolling(window=window, min_periods=1).max()
    return PanelContainer(result_data)

def op_ts_min(data: PanelContainer, window: int) -> PanelContainer:
    """计算面板数据在时间序列上的滚动最小值。"""
    if not isinstance(data, PanelContainer):
        raise TypeError("操作数 'data' 必须是 PanelContainer。")
    if not isinstance(window, int) or window <= 0:
        raise TypeError("参数 'window' 必须是正整数。")
    result_data = data.get_data().rolling(window=window, min_periods=1).min()
    return PanelContainer(result_data)

def op_ts_rank(data: PanelContainer, window: int) -> PanelContainer:
    """计算面板数据在时间序列上的滚动排名。"""
    if not isinstance(data, PanelContainer):
        raise TypeError("操作数 'data' 必须是 PanelContainer。")
    if not isinstance(window, int) or window <= 0:
        raise TypeError("参数 'window' 必须是正整数。")
    # apply rank on a rolling window. The last element of the window is ranked against the elements in the window.
    result_data = data.get_data().rolling(window=window).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    return PanelContainer(result_data)
    

# --- Cross-Sectional Operators ---

def op_cs_rank(data: PanelContainer) -> PanelContainer:
    """在每个时间点上（行）对数据进行横截面排名。"""
    if not isinstance(data, PanelContainer):
        raise TypeError("操作数 'data' 必须是 PanelContainer。")
    result_data = data.get_data().rank(axis=1, pct=True)
    return PanelContainer(result_data)

def op_cs_normalize(data: PanelContainer) -> PanelContainer:
    """在每个时间点上（行）对数据进行横截面 Z-score 标准化。"""
    if not isinstance(data, PanelContainer):
        raise TypeError("操作数 'data' 必须是 PanelContainer。")
    df = data.get_data()
    mean = df.mean(axis=1)
    std = df.std(axis=1)
    # df.sub(mean, axis=0) -> subtracts the mean of each row from each element in that row
    # df.div(std, axis=0) -> divides each element in each row by the std of that row
    result_data = df.sub(mean, axis=0).div(std, axis=0)
    result_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    result_data.fillna(0, inplace=True)
    return PanelContainer(result_data) 