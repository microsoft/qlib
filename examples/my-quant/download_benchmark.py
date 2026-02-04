#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.
"""
下载基准指数数据 (沪深300)

使用方法:
    python download_benchmark.py
"""
import sys
from pathlib import Path

import akshare as ak
import pandas as pd
from loguru import logger


def download_benchmark_data(
    save_dir="D:/Quant-qlib-official/data/source",
    start_date="20080101",
    end_date="20250101",
):
    """
    下载沪深300指数数据并保存

    Parameters
    ----------
    save_dir : str
        保存目录
    start_date : str
        开始日期，格式 YYYYMMDD
    end_date : str
        结束日期，格式 YYYYMMDD
    """
    save_dir = Path(save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)

    # 获取沪深300指数日线数据
    logger.info("获取沪深300指数数据...")
    df = ak.stock_zh_index_daily(symbol="sh000300")

    # 转换日期格式
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

    # 过滤日期范围
    df = df[(df['date'] >= start_date[:4] + '-' + start_date[4:6] + '-' + start_date[6:]) &
            (df['date'] <= end_date[:4] + '-' + end_date[4:6] + '-' + end_date[6:])]

    # 添加 symbol 列
    df['symbol'] = 'SH000300'

    # 重命名列以匹配 qlib 格式
    # 注意：指数数据没有 volume，使用 close * 1000000 作为 money
    df['money'] = df['close'] * 1000000  # 指数没有成交量，用估算值
    df['volume'] = 0  # 指数没有成交量
    df['outstanding_share'] = 1  # 指数没有流通股
    df['turnover'] = 0
    df['paused'] = 0
    df['pct_chg'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1) * 100

    # 计算 vwap
    df['vwap'] = df['close']

    # 选择需要的列
    columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'money',
               'outstanding_share', 'turnover', 'symbol', 'pct_chg', 'vwap', 'paused']
    df = df[columns]

    # 保存
    save_path = save_dir / "SH000300.csv"
    df.to_csv(save_path, index=False)
    logger.info(f"已保存基准数据: {save_path}")
    logger.info(f"共 {len(df)} 条数据，日期范围: {df['date'].min()} 到 {df['date'].max()}")

    return df


def main():
    """主函数"""
    download_benchmark_data()


if __name__ == "__main__":
    main()
