import pandas as pd

class Calendar:
    """
    提供交易日历服务。
    对于这个初始版本，它只生成一个营业日范围。
    """
    def __init__(self, start_date: str, end_date: str):
        # 使用 'B' 表示营业日频率，这是交易日的常用替代。
        self._dates = pd.date_range(start=start_date, end=end_date, freq='B')

    def get_trading_days(self, start_date: str = None, end_date: str = None) -> pd.DatetimeIndex:
        """
        返回指定范围内的交易日。
        如果未提供 start_date 或 end_date，则使用日历的默认范围。
        """
        if start_date is None and end_date is None:
            return self._dates
        
        start = pd.to_datetime(start_date) if start_date else self._dates[0]
        end = pd.to_datetime(end_date) if end_date else self._dates[-1]

        return self._dates[(self._dates >= start) & (self._dates <= end)]

    def __repr__(self):
        return f"Calendar(start={self._dates[0].date()}, end={self._dates[-1].date()}, days={len(self._dates)})" 