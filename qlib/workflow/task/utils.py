# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import bisect
import pandas as pd
from qlib.data import D
from qlib.config import C
from qlib.log import get_module_logger
from pymongo import MongoClient
from typing import Union


def get_mongodb():
    """

    get database in MongoDB, which means you need to declare the address and the name of database.
    for example:
        C["mongo"] = {
            "task_url" : "mongodb://localhost:27017/",
            "task_db_name" : "rolling_db"
        }

    """
    try:
        cfg = C["mongo"]
    except KeyError:
        get_module_logger("task").error("Please configure `C['mongo']` before using TaskManager")
        raise

    client = MongoClient(cfg["task_url"])
    return client.get_database(name=cfg["task_db_name"])


class TimeAdjuster:
    """
    find appropriate date and adjust date.
    """

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
        (Deprecated)
        Return the max calendar datetime
        """
        return max(self.cals)

    def last_date(self) -> pd.Timestamp:
        """
        Return the last datetime in the calendar
        """
        return self.cals[-1]

    def align_idx(self, time_point, tp_type="start"):
        """
        align the index of time_point in the calendar

        Parameters
        ----------
        time_point
        tp_type : str

        Returns
        -------
        index : int
        """
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
        Align time_point to trade date of calendar

        Parameters
        ----------
        time_point
            Time point
        tp_type : str
            time point type (`"start"`, `"end"`)
        """
        return self.cals[self.align_idx(time_point, tp_type=tp_type)]

    def align_seg(self, segment: Union[dict, tuple]):
        """
        align the given date to trade date

        for example:
            input: {'train': ('2008-01-01', '2014-12-31'), 'valid': ('2015-01-01', '2016-12-31'), 'test': ('2017-01-01', '2020-08-01')}

            output: {'train': (Timestamp('2008-01-02 00:00:00'), Timestamp('2014-12-31 00:00:00')),
                    'valid': (Timestamp('2015-01-05 00:00:00'), Timestamp('2016-12-30 00:00:00')),
                    'test': (Timestamp('2017-01-03 00:00:00'), Timestamp('2020-07-31 00:00:00'))}

        Parameters
        ----------
        segment

        Returns
        -------
        the start and end trade date (pd.Timestamp) between the given start and end date.
        """
        if isinstance(segment, dict):
            return {k: self.align_seg(seg) for k, seg in segment.items()}
        elif isinstance(segment, tuple):
            return self.align_time(segment[0], tp_type="start"), self.align_time(segment[1], tp_type="end")
        else:
            raise NotImplementedError(f"This type of input is not supported")

    def truncate(self, segment: tuple, test_start, days: int):
        """
        truncate the segment based on the test_start date

        Parameters
        ----------
        segment : tuple
            time segment
        test_start
        days : int
            The trading days to be truncated
            the data in this segment may need 'days' data
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

    def shift(self, seg: tuple, step: int, rtype=SHIFT_SD):
        """
        shift the datatime of segment

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
