from __future__ import annotations
from enum import Enum

from typing import List, Tuple


class FinancialInterval(Enum):
    YEARLY = 1
    QUARTERLY = 4

    @classmethod
    def from_alias(cls, alias) -> FinancialInterval:
        # TODO: Use the same variable as the `to_alias` method
        alias_map = {"y": cls.YEARLY, "q": cls.QUARTERLY}
        return alias_map[alias]

    def to_alias(self) -> str:
        alias_map = {self.YEARLY: "y", self.QUARTERLY: "q"}
        return alias_map[self]

    def get_period_list(self, start_period: int, end_period: int) -> List[int]:
        if self == self.YEARLY:
            return list(range(start_period, end_period + 1))
        elif self == self.QUARTERLY:
            return [
                year * 100 + quarter
                for year in range(start_period // 100, end_period // 100 + 1)
                for quarter in range(1, 5)
                if start_period <= (year * 100 + quarter) <= end_period
            ]
        else:
            raise ValueError(f"{self} is not supported.")

    def get_period_offset(self, start_year: int, period: int) -> int:
        if self == self.YEARLY:
            rv = period - start_year
        elif self == self.QUARTERLY:
            rv = (period // 100 - start_year) * 4 + period % 100 - 1
        else:
            raise ValueError(f"{self} is not supported.")
        rv = max(rv, 0)
        return rv

    def get_period_slice(self, s: slice) -> Tuple[int, int]:
        from qlib.data.data import Cal  # pylint: disable=C0415

        _calendar = Cal.calendar(freq="day")
        try:
            start_time = _calendar[s.start]
            end_time = _calendar[s.stop - 2]
        except IndexError as e:
            raise ValueError("Index out of calendar range") from e

        if self == self.YEARLY:
            return start_time.year, end_time.year
        elif self == self.QUARTERLY:

            def quarterly_func(x) -> int:
                return int(x.year) * 100 + int(x.quarter)

            return quarterly_func(start_time), quarterly_func(end_time)
        else:
            raise ValueError(f"{self} is not supported.")

    def get_year(self, period: int) -> int:
        if self == self.YEARLY:
            return period
        elif self == self.QUARTERLY:
            return period // 100
        else:
            raise ValueError(f"{self} is not supported.")
