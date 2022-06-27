from typing import List

from qlib.backtest.executor import NestedExecutor
from .strategy import RLStrategyBase


class RLNestedExecutor(NestedExecutor):
    # RL nested executor
    def post_inner_exe_step(self, inner_exe_res: List[object]) -> None:
        if isinstance(self.inner_strategy, RLStrategyBase):
            self.inner_strategy.post_exe_step(inner_exe_res)
