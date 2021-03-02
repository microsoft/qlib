# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import bisect
import pandas as pd
from qlib.data import D
from qlib.config import C
from qlib.log import get_module_logger
from pymongo import MongoClient


def get_mongodb():
    try:
        cfg = C["mongo"]
    except KeyError:
        get_module_logger("task").error("Please configure `C['mongo']` before using TaskManager")
        raise

    client = MongoClient(cfg["task_url"])
    return client.get_database(name=cfg["task_db_name"])


class TimeAdjuster:
    """找到合适的日期，然后adjust date"""

    def __init__(self, future=False):
        self.cals = D.calendar(future=future)

    def get(self, idx: int):
        """
        Get datetime by index

        Parameters
        ----------
        idx : int
            index of the calendar
        """
        if idx >= len(self.cals):
            return None
        return self.cals[idx]

    def max(self):
        """
        Return return the max calendar date
        """
        return max(self.cals)

    def align_idx(self, time_point, tp_type="start"):
        time_point = pd.Timestamp(time_point)
        if tp_type == "start":
            idx = bisect.bisect_left(self.cals, time_point)
        elif tp_type == "end":
            idx = bisect.bisect_right(self.cals, time_point) - 1
        else:
            raise NotImplementedError(f"This type of input is not supported")
        return idx

    def align_time(self, time_point, tp_type="start"):
        """
        Align a timepoint to calendar  weekdays

        Parameters
        ----------
        time_point :
            Time point
        tp_type : str
            time point type (`"start"`, `"end"`)
        """
        return self.cals[self.align_idx(time_point, tp_type=tp_type)]

    def align_seg(self, segment):
        if isinstance(segment, dict):
            return {k: self.align_seg(seg) for k, seg in segment.items()}
        elif isinstance(segment, tuple):
            return self.align_time(segment[0], tp_type="start"), self.align_time(segment[1], tp_type="end")
        else:
            raise NotImplementedError(f"This type of input is not supported")

    def truncate(self, segment, test_start, days: int):
        """
        truncate the segment based on the test_start date

        Parameters
        ----------
        segment :
            time segment
        days : int
            The trading days to be truncated
            大部分情况是因为这个时间段的数据(一般是特征)会用到 `days` 天的数据
        """
        test_idx = self.align_idx(test_start)
        if isinstance(segment, tuple):
            new_seg = []
            for time_point in segment:
                tp_idx = min(self.align_idx(time_point), test_idx - days)
                assert tp_idx > 0
                new_seg.append(self.get(tp_idx))
            return tuple(new_seg)
        else:
            raise NotImplementedError(f"This type of input is not supported")

    SHIFT_SD = "sliding"
    SHIFT_EX = "expanding"

    def shift(self, seg, step: int, rtype=SHIFT_SD):
        """
        shift the datatiem of segment

        Parameters
        ----------
        seg :
            datetime segment
        step : int
            rolling step
        rtype : str
            rolling type ("sliding" or "expanding")

        Raises
        ------
        KeyError:
            shift will raise error if the index(both start and end) is out of self.cal
        """
        if isinstance(seg, tuple):
            start_idx, end_idx = self.align_idx(seg[0], tp_type="start"), self.align_idx(seg[1], tp_type="end")
            if rtype == self.SHIFT_SD:
                start_idx += step
                end_idx += step
            elif rtype == self.SHIFT_EX:
                end_idx += step
            else:
                raise NotImplementedError(f"This type of input is not supported")
            if start_idx > len(self.cals):
                raise KeyError("The segment is out of valid calendar")
            return self.get(start_idx), self.get(end_idx)
        else:
            raise NotImplementedError(f"This type of input is not supported")
