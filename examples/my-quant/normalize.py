#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.
"""
数据规范化脚本

功能:
1. 获取交易日历
2. 对齐所有股票数据到交易日历
3. 处理涨跌停、停牌等特殊情况
4. 保存规范化后的数据

使用方法:
    python normalize.py --source_dir ~/.qlib/akshare_data/source --target_dir ~/.qlib/akshare_data/normalize
"""
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

import fire
import numpy as np
import pandas as pd
from loguru import logger

try:
    import akshare as ak
except ImportError:
    logger.error("请先安装akshare: pip install akshare")
    sys.exit(1)


def get_trading_calendar(start_date="2008-01-01", end_date="2025-12-31"):
    """
    获取交易日历

    Parameters
    ----------
    start_date : str
        开始日期
    end_date : str
        结束日期

    Returns
    -------
    list
        交易日列表
    """
    logger.info("获取交易日历...")
    trade_date_df = ak.tool_trade_date_hist_sina()

    # 筛选日期范围
    trade_date_df['trade_date'] = pd.to_datetime(trade_date_df['trade_date'])
    mask = (trade_date_df['trade_date'] >= start_date) & (trade_date_df['trade_date'] <= end_date)
    calendar_list = trade_date_df.loc[mask, 'trade_date'].tolist()

    logger.info(f"获取到 {len(calendar_list)} 个交易日")
    return sorted(calendar_list)

# 这是填充空值的逻辑

# def normalize_stock_data(df, calendar_list, symbol_field_name='symbol', date_field_name='date'):
#     """
#     规范化单只股票数据

#     Parameters
#     ----------
#     df : pd.DataFrame
#         原始数据
#     calendar_list : list
#         交易日历
#     symbol_field_name : str
#         股票代码字段名
#     date_field_name : str
#         日期字段名

#     Returns
#     -------
#     pd.DataFrame
#         规范化后的数据
#     """
#     if df is None or df.empty:
#         return None

#     symbol = df[symbol_field_name].iloc[0] if symbol_field_name in df.columns else None
#     if symbol is None:
#         logger.warning("无法获取股票代码")
#         return None

#     df = df.copy()

#     # 确保日期列是datetime格式
#     if date_field_name in df.columns:
#         df[date_field_name] = pd.to_datetime(df[date_field_name])
#         df = df.set_index(date_field_name)
#     else:
#         logger.warning(f"找不到日期列: {date_field_name}")
#         return None

#     # 删除重复日期
#     df = df[~df.index.duplicated(keep='first')]

#     # 只保留交易日历中股票实际存在的日期范围
#     data_min_date = df.index.min()
#     data_max_date = df.index.max()
#     calendar_index = pd.DatetimeIndex([d for d in calendar_list if data_min_date <= d <= data_max_date])

#     # 重新索引到交易日历（只包含股票实际存在的日期范围）
#     df = df.reindex(calendar_index)

#     # 恢复symbol列
#     df[symbol_field_name] = symbol

#     # 计算涨跌幅(pct_chg列可能已有)
#     if 'pct_chg' not in df.columns and 'close' in df.columns:
#         df['pct_chg'] = df['close'].pct_change() * 100

#     # 处理停牌数据
#     # 停牌时，成交量为0或NaN
#     if 'volume' in df.columns:
#         df['paused'] = ((df['volume'] == 0) | (df['volume'].isna())).astype(int)

#     # 重置索引，将date变回列
#     df = df.reset_index()
#     df = df.rename(columns={'index': date_field_name})

#     # 添加factor列（默认1.0，因为数据已经是后复权）
#     # qlib数据处理需要此列，但我们的数据不需要调整
#     if 'factor' not in df.columns:
#         df['factor'] = 1.0

#     return df


# ffill的逻辑
def normalize_stock_data(df, calendar_list, symbol_field_name='symbol', date_field_name='date'):
    """
    规范化单只股票数据 - 修复版
    """
    if df is None or df.empty:
        return None

    symbol = df[symbol_field_name].iloc[0] if symbol_field_name in df.columns else None
    if symbol is None:
        logger.warning("无法获取股票代码")
        return None

    df = df.copy()

    # 1. 确保日期格式并设为索引
    if date_field_name in df.columns:
        df[date_field_name] = pd.to_datetime(df[date_field_name])
        df = df.set_index(date_field_name)
    else:
        logger.warning(f"找不到日期列: {date_field_name}")
        return None

    # 2. 去重
    df = df[~df.index.duplicated(keep='first')]

    # 3. 确定时间范围（只处理上市后的时间段）
    data_min_date = df.index.min()
    data_max_date = df.index.max()
    
    # 筛选出该股票上市期间的标准交易日历
    calendar_index = pd.DatetimeIndex([d for d in calendar_list if data_min_date <= d <= data_max_date])

    # 4. 【核心修改】Reindex 会引入 NaN，必须在这一步之后处理
    df = df.reindex(calendar_index)

    # 恢复symbol列 (reindex后可能会丢失或变成NaN)
    df[symbol_field_name] = symbol

    # 5. 【关键修复步骤】填充停牌日的 NaN 数据
    
    # A. 价格类字段：使用前向填充 (ffill)
    # 逻辑：停牌期间，价格等于停牌前最后一天的收盘价
    price_cols = ['close', 'open', 'high', 'low', 'adj_close', 'outstanding_share'] # 根据你实际列名增减
    target_cols = [c for c in price_cols if c in df.columns]
    if target_cols:
        df[target_cols] = df[target_cols].ffill()

    # B. 成交量类字段：填充为 0
    # 逻辑：停牌期间没有成交
    vol_cols = ['volume', 'amount', 'turnover', 'money'] 
    for c in vol_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    # C. 处理因子列 (Factor)
    # 如果数据源本身有 factor，也需要 ffill，否则后续计算会出错
    if 'factor' in df.columns:
        df['factor'] = df['factor'].ffill()
    else:
        # 如果没有，默认为 1.0 (全填充)
        df['factor'] = 1.0

    # 6. 二次清洗：去除仍然含有 NaN 的行
    # ffill 无法处理上市首日如果不在 calendar 开头的情况（虽然概率很低，但为了 Qlib 不报错）
    df = df.dropna(subset=['close'])

    # 7. 计算衍生字段
    if 'pct_chg' not in df.columns and 'close' in df.columns:
        # 注意：这里计算 pct_chg 会因为 ffill 导致停牌日为 0，这是正确的
        df['pct_chg'] = df['close'].pct_change() * 100

    # 标记停牌 (此时 volume 已经是 0 了)
    if 'volume' in df.columns:
        df['paused'] = (df['volume'] == 0).astype(int)

    # 8. 重置索引
    df = df.reset_index()
    df = df.rename(columns={'index': date_field_name})

    return df



def process_file(file_path, calendar_list, source_dir, target_dir):
    """
    处理单个文件

    Parameters
    ----------
    file_path : Path
        源文件路径
    calendar_list : list
        交易日历
    source_dir : Path
        源目录
    target_dir : Path
        目标目录

    Returns
    -------
    str
        处理的文件名
    """
    try:
        # 读取CSV
        df = pd.read_csv(file_path, low_memory=False)

        # 规范化
        normalized_df = normalize_stock_data(df, calendar_list)

        if normalized_df is not None and not normalized_df.empty:
            # 保存到目标目录
            target_path = target_dir / file_path.name
            normalized_df.to_csv(target_path, index=False)
            return str(file_path.name)
    except Exception as e:
        logger.warning(f"处理 {file_path.name} 失败: {e}")

    return None


def normalize_all(
    source_dir="D:/Quant-qlib-official/data/source",
    target_dir="D:/Quant-qlib-official/data/normalize",
    start_date="2008-01-01",
    end_date="2025-12-31",
    max_workers=16
):
    """
    规范化所有数据

    Parameters
    ----------
    source_dir : str
        源数据目录
    target_dir : str
        目标数据目录
    start_date : str
        开始日期
    end_date : str
        结束日期
    max_workers : int
        并行处理线程数
    """
    source_dir = Path(source_dir).expanduser()
    target_dir = Path(target_dir).expanduser()

    # 创建目标目录
    target_dir.mkdir(parents=True, exist_ok=True)

    # 获取交易日历
    calendar_list = get_trading_calendar(start_date, end_date)

    # 获取文件列表
    file_list = list(source_dir.glob("*.csv"))
    logger.info(f"找到 {len(file_list)} 个文件需要处理")

    success_count = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_file,
                file_path,
                calendar_list,
                source_dir,
                target_dir
            )
            for file_path in file_list
        ]

        for future in tqdm(futures, total=len(futures), desc="规范化进度"):
            result = future.result()
            if result:
                success_count += 1

    logger.info(f"规范化完成: 成功处理 {success_count}/{len(file_list)} 个文件")
    return success_count


def main():
    """主函数"""
    fire.Fire({
        'normalize': normalize_all,
    })


if __name__ == "__main__":
    main()
