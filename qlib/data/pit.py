# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Qlib follow the logic below to supporting point-in-time database

For each stock, the format of its data is <observe_time, feature>. Expression Engine support calculation on such format of data

To calculate the feature value f_t at a specific observe time t,  data with format <period_time, feature> will be used.
For example, the average earning of last 4 quarters (period_time) on 20190719 (observe_time)

The calculation of both <period_time, feature> and <observe_time, feature> data rely on expression engine. It consists of 2 phases.
1) calculation <period_time, feature> at each observation time t and it will collasped into a point (just like a normal feature)
2) concatenate all th collasped data, we will get data with format <observe_time, feature>.
Qlib will use the operator `P` to perform the collapse.
"""
from typing import Tuple, Any, Optional

import numpy as np
import pandas as pd
from qlib.data.ops import ElemOperator
from qlib.log import get_module_logger
from .data import Cal


class P(ElemOperator):
    def get_observe_data(
        self, instrument: str, start_index: int, end_index: int, freq: str, period: Optional[int] = None
    ) -> pd.Series:
        # Observe time may populate values with data from the reporting period prior to the end date
        series = self.feature.load(instrument, 0, end_index, freq)
        _calendar = Cal.calendar(freq=freq)
        resample_data = np.empty(end_index - start_index + 1, dtype="float32")
        for time_idx in range(start_index, end_index + 1):
            current_time = _calendar[time_idx]
            try:
                s = series[series.index.get_level_values("datetime") <= current_time]
                if period is not None:
                    s = s[s.index.get_level_values("period") == period]
                resample_data[time_idx - start_index] = s.iloc[-1] if len(s) > 0 else np.nan
            except FileNotFoundError:
                get_module_logger("base").warning(f"WARN: period data not found for {str(self)}")
                return pd.Series(dtype="float32", name=str(self))

        resample_series = pd.Series(
            resample_data, index=pd.RangeIndex(start_index, end_index + 1), dtype="float32", name=str(self)
        )
        return resample_series

    def _load_internal(self, instrument: str, start_index: int, end_index: int, *args: Tuple[Any]) -> pd.Series:
        return self.get_observe_data(instrument, start_index, end_index, *args)

    def get_longest_back_rolling(self) -> int:
        # The period data will collapse as a normal feature. So no extending and looking back
        return 0

    def get_extended_window_size(self) -> Tuple[int, int]:
        # The period data will collapse as a normal feature. So no extending and looking back
        return 0, 0


class PRef(P):
    def __init__(self, feature, period) -> None:
        super().__init__(feature)
        self.period = period

    def __str__(self) -> str:
        return f"{super().__str__()}[{self.period}]"

    def _load_internal(self, instrument: str, start_index: int, end_index: int, *args: Tuple[Any]) -> pd.Series:
        return self.get_observe_data(instrument, start_index, end_index, *args, period=self.period)
