from abc import ABC, abstractmethod
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from typing import List, Dict, Optional

from .containers import PanelContainer
from .calendar import Calendar

class DataProvider(ABC):
    """数据提供者的抽象基类。"""
    @abstractmethod
    def load(self, field: str, start_date: str, end_date: str, stocks: Optional[List[str]] = None) -> PanelContainer:
        """
        加载指定字段、时间范围和股票列表的数据。
        """
        pass

class ParquetDataProvider(DataProvider):
    """
    从磁盘加载 Parquet 数据，支持按需转换为宽格式并进行缓存。
    预期 Parquet 文件格式：/data_path/{YYYY}/{YYYYMMDD}.parquet
    文件内包含 'date', 'code', 和其他因子字段。
    """
    def __init__(self, data_path: str, calendar: Calendar):
        self._data_path = Path(data_path)
        self._calendar = calendar
        self._cache: Dict[str, pd.DataFrame] = {}

    def load(self, field: str, start_date: str, end_date: str, stocks: Optional[List[str]] = None) -> PanelContainer:
        """
        加载给定字段和时间范围内的数据。
        
        1. 检查缓存 (基于字段和日期范围生成缓存键)。
        2. 根据日历确定所需文件路径。
        3. 读取并合并 Parquet 文件。
        4. 将长格式 DataFrame 转换为宽格式。
        5. 如果指定了股票列表，则进行筛选。
        6. 缓存结果并返回一个 PanelContainer。
        """
        cache_key = f"{field}_{start_date}_{end_date}"
        if cache_key in self._cache:
            cached_df = self._cache[cache_key]
            if stocks:
                # 确保只选择存在的列
                valid_stocks = [s for s in stocks if s in cached_df.columns]
                cached_df = cached_df[valid_stocks]
            return PanelContainer(cached_df)

        trading_days = self._calendar.get_trading_days(start_date, end_date)
        file_paths = self._get_file_paths(trading_days)
        
        if not file_paths:
            raise FileNotFoundError(f"在指定日期范围 {start_date} - {end_date} 内没有找到数据文件。")

        all_data = []
        found_files = 0
        columns_to_read = ['date', 'code', field]
        
        for path in file_paths:
            if path.exists():
                table = pq.read_table(path, columns=columns_to_read)
                all_data.append(table.to_pandas())
                found_files += 1
        
        if found_files == 0:
            raise FileNotFoundError(f"虽然生成了文件路径，但在文件系统中并未找到任何 '{field}' 的有效数据文件。")
            
        long_df = pd.concat(all_data, ignore_index=True)
        
        long_df['date'] = pd.to_datetime(long_df['date'])

        wide_df = long_df.pivot(index='date', columns='code', values=field)
        
        # 缓存完整加载的结果 (不按股票筛选)
        self._cache[cache_key] = wide_df
        
        if stocks:
            # 确保只选择存在的列
            valid_stocks = [s for s in stocks if s in wide_df.columns]
            wide_df = wide_df[valid_stocks]

        return PanelContainer(wide_df)

    def _get_file_paths(self, dates: pd.DatetimeIndex) -> List[Path]:
        """
        根据文档示例构建文件路径: /.../{YYYY}/{YYYYMMDD}.parquet
        """
        paths = []
        for date in dates:
            year = date.year
            yyyymmdd = date.strftime('%Y%m%d')
            path = self._data_path / str(year) / f"{yyyymmdd}.parquet"
            paths.append(path)
        return list(dict.fromkeys(paths)) 