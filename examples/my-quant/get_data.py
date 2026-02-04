#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.
"""
使用akshare获取A股日线数据

使用方法:
    python get_data.py --start_date 20080101 --end_date 20250101 --max_workers 1
"""
import sys
import time
import fire
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import numpy as np
import pandas as pd
from loguru import logger

# 导入akshare
try:
    import akshare as ak
except ImportError:
    logger.error("请先安装akshare: pip install akshare")
    sys.exit(1)


def get_stock_list():
    """获取全市场A股列表"""
    logger.info("获取全市场A股列表...")
    df = ak.stock_info_a_code_name()
    # 获取所有股票代码，添加交易所前缀
    # 上海: 6开头 -> sh
    # 深圳: 0、3开头 -> sz
    # 8/9开头: 跳过（北交所、配售股票等特殊类型）
    symbols = []
    skipped = 0
    for code in df['code']:
        if code.startswith('6'):
            symbols.append(f'sh{code}')
        elif code.startswith(('8', '9')):
            skipped += 1  # 北交所、配售股票等跳过
        else:
            symbols.append(f'sz{code}')
    logger.info(f"获取到 {len(symbols)} 只股票，跳过 {skipped} 只特殊类型股票")
    return symbols


def get_daily_data(symbol, start_date, end_date):
    """
    获取单只股票日线数据

    Parameters
    ----------
    symbol : str
        股票代码，如 "sh600519" 或 "600519"
    start_date : str
        开始日期，格式 YYYYMMDD
    end_date : str
        结束日期，格式 YYYYMMDD

    Returns
    -------
    pd.DataFrame
        日线数据
    """
    try:
        # akshare接口
        df = ak.stock_zh_a_daily(symbol=symbol, start_date=start_date, end_date=end_date, adjust='hfq')

        if df is None or df.empty:
            return None

        # 重置索引，date变成列
        df = df.reset_index(drop=True)

        # # 添加symbol列（保留6位数字格式）
        # code = symbol.replace('sh', '').replace('sz', '')
        # df['symbol'] = str(code).zfill(6)  # 保持6位数字格式
        df['symbol'] = symbol.upper()

        # 转换数据类型
        for col in ['open', 'close', 'high', 'low', 'volume', 'amount', 'pct_chg', 'change', 'turnover']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 重命名 amount 为 money，适配qlib格式
        if 'amount' in df.columns:
            df = df.rename(columns={'amount': 'money'})

        return df

    except Exception as e:
        logger.warning(f"获取 {symbol} 数据失败: {e}")
        return None


def save_to_csv(df, save_dir, symbol):
    """保存为CSV格式"""
    if df is None or df.empty:
        return False

    # # 使用纯数字代码作为文件名
    # code = symbol.replace('sh', '').replace('sz', '')
    # save_path = Path(save_dir) / f"{code}.csv"
    file_name = symbol.upper() # 变成 SH600519
    save_path = Path(save_dir) / f"{file_name}.csv"

    # 确保date和symbol列是字符串格式
    df['date'] = df['date'].astype(str)
    df['symbol'] = df['symbol'].astype(str)

    # 如果文件已存在，合并数据
    if save_path.exists():
        old_df = pd.read_csv(save_path, dtype={'symbol': str})
        # 合并并去重
        df = pd.concat([old_df, df], ignore_index=True)
        df = df.drop_duplicates(subset=['date', 'symbol'], keep='last')
        df = df.sort_values('date')

    df.to_csv(save_path, index=False)
    return True


def download_stock_data(
    symbol,
    save_dir,
    start_date,
    end_date,
    delay=0.5,
    max_retries=3
):
    """
    下载单只股票数据，包含重试机制

    Parameters
    ----------
    symbol : str
        股票代码
    save_dir : str
        保存目录
    start_date : str
        开始日期
    end_date : str
        结束日期
    delay : float
        请求间隔
    max_retries : int
        最大重试次数

    Returns
    -------
    bool
        是否成功
    """
    for attempt in range(max_retries):
        try:
            time.sleep(delay)
            df = get_daily_data(symbol, start_date, end_date)

            if df is not None and not df.empty:
                save_to_csv(df, save_dir, symbol)
                return True
            else:
                return False

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)  # 失败后等待更长时间
                continue
            logger.warning(f"下载 {symbol} 失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            return False


def download_all_data(
    save_dir="D:/Quant-qlib-official/data/source",
    start_date="20080101",
    end_date="20250101",
    max_workers=12,
    delay=0.5,
    limit_nums=None,
    exists_skip=False
):
    """
    下载所有股票数据

    Parameters
    ----------
    save_dir : str
        数据保存目录
    start_date : str
        开始日期
    end_date : str
        结束日期
    max_workers : int
        并行下载线程数
    delay : float
        请求间隔
    limit_nums : int
        限制下载数量（用于测试）
    exists_skip : bool
        已存在的文件是否跳过
    """
    save_dir = Path(save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)

    # 获取股票列表
    symbols = get_stock_list()

    if limit_nums:
        symbols = symbols[:limit_nums]
        logger.info(f"测试模式: 只下载 {limit_nums} 只股票")

    logger.info(f"开始下载 {len(symbols)} 只股票的数据...")

    success_count = 0
    fail_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                download_stock_data,
                symbol,
                save_dir,
                start_date,
                end_date,
                delay,
            ): symbol for symbol in symbols
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="下载进度"):
            symbol = futures[future]
            try:
                result = future.result()
                if result:
                    success_count += 1
                else:
                    fail_count += 1
            except Exception as e:
                logger.warning(f"处理 {symbol} 时出错: {e}")
                fail_count += 1

    logger.info(f"下载完成: 成功 {success_count}, 失败 {fail_count}")
    return success_count, fail_count


def main():
    """主函数"""
    fire.Fire({
        'download': download_all_data,
    })


if __name__ == "__main__":
    main()
