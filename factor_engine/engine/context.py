from typing import List, Optional

class ExecutionContext:
    """一个简单的数据类，用于在执行期间清晰地传递上下文信息。"""
    def __init__(self, start_date: str, end_date: str, stocks: Optional[List[str]] = None):
        self.start_date = start_date
        self.end_date = end_date
        self.stocks = stocks 