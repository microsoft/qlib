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

import numpy as np
import pandas as pd
from qlib.data.ops import ElemOperator
from qlib.log import get_module_logger
from .data import Cal


def _is_calendar_position(value):
    """Return whether ``value`` is an integer calendar index.

    ``bool`` is a subclass of ``int``, but it is not a calendar position.
    """
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


class P(ElemOperator):
    """Collapse a point-in-time feature onto each observation time.

    Parameters
    ----------
    feature : Expression
        PIT feature, for example ``$$roewa_q``. At each observation time the
        operator keeps the latest value published at or before that time.

    Notes
    -----
    With ``time2idx=True`` (the default) ``start_index`` and ``end_index`` are
    integer calendar positions and the result keeps a ``RangeIndex``.
    With ``time2idx=False`` they are the query datetimes. Those bounds are
    aligned to the trading calendar, and the result uses that datetime index
    so ``series.loc[start_time:end_time]`` still selects the requested range.
    """

    def _load_internal(self, instrument, start_index, end_index, freq):
        calendar = Cal.calendar(freq=freq)
        datetime_query = not (_is_calendar_position(start_index) and _is_calendar_position(end_index))
        if datetime_query:
            # Timestamp - Timestamp is a Timedelta, so the length expression
            # below is only valid after the bounds are calendar positions.
            _, _, start_index, end_index = Cal.locate_index(start_index, end_index, freq=freq, future=False)
        start_index = int(start_index)
        end_index = int(end_index)
        if datetime_query and end_index < start_index:
            return pd.Series(dtype="float32", name=str(self))

        resample_data = np.empty(end_index - start_index + 1, dtype="float32")

        for cur_index in range(start_index, end_index + 1):
            cur_time = calendar[cur_index]
            # To load expression accurately, more historical data are required
            start_ws, end_ws = self.feature.get_extended_window_size()
            if end_ws > 0:
                raise ValueError(
                    "PIT database does not support referring to future period (e.g. expressions like `Ref('$$roewa_q', -1)` are not supported"
                )

            # The calculated value will always the last element, so the end_offset is zero.
            try:
                s = self._load_feature(instrument, -start_ws, 0, cur_time)
                resample_data[cur_index - start_index] = s.iloc[-1] if len(s) > 0 else np.nan
            except FileNotFoundError:
                get_module_logger("base").warning(f"WARN: period data not found for {str(self)}")
                return pd.Series(dtype="float32", name=str(self))

        if datetime_query:
            index = pd.Index(calendar[start_index : end_index + 1])
        else:
            index = pd.RangeIndex(start_index, end_index + 1)
        resample_series = pd.Series(resample_data, index=index, dtype="float32", name=str(self))
        return resample_series

    def _load_feature(self, instrument, start_index, end_index, cur_time):
        return self.feature.load(instrument, start_index, end_index, cur_time)

    def get_longest_back_rolling(self):
        # The period data will collapse as a normal feature. So no extending and looking back
        return 0

    def get_extended_window_size(self):
        # The period data will collapse as a normal feature. So no extending and looking back
        return 0, 0


class PRef(P):
    def __init__(self, feature, period):
        super().__init__(feature)
        self.period = period

    def __str__(self):
        return f"{super().__str__()}[{self.period}]"

    def _load_feature(self, instrument, start_index, end_index, cur_time):
        return self.feature.load(instrument, start_index, end_index, cur_time, self.period)
