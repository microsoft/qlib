# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Some tools for task management.
"""

import bisect
import pandas as pd
from qlib.data import D
from qlib.workflow import R
from qlib.config import C
from qlib.log import get_module_logger
from pymongo import MongoClient
from pymongo.database import Database
from typing import Union


def get_mongodb() -> Database:

    """
    Get database in MongoDB, which means you need to declare the address and the name of a database at first.

    For example:

        Using qlib.init():

            mongo_conf = {
                "task_url": task_url,  # your MongoDB url
                "task_db_name": task_db_name,  # database name
            }
            qlib.init(..., mongo=mongo_conf)

        After qlib.init():

            C["mongo"] = {
                "task_url" : "mongodb://localhost:27017/",
                "task_db_name" : "rolling_db"
            }

    Returns:
        Database: the Database instance
    """
    try:
        cfg = C["mongo"]
    except KeyError:
        get_module_logger("task").error("Please configure `C['mongo']` before using TaskManager")
        raise

    client = MongoClient(cfg["task_url"])
    return client.get_database(name=cfg["task_db_name"])


def list_recorders(experiment, rec_filter_func=None):
    """
    List all recorders which can pass the filter in an experiment.

    Args:
        experiment (str or Experiment): the name of an Experiment or an instance
        rec_filter_func (Callable, optional): return True to retain the given recorder. Defaults to None.

    Returns:
        dict: a dict {rid: recorder} after filtering.
    """
    if isinstance(experiment, str):
        experiment = R.get_exp(experiment_name=experiment)
    recs = experiment.list_recorders()
    recs_flt = {}
    for rid, rec in recs.items():
        if rec_filter_func is None or rec_filter_func(rec):
            recs_flt[rid] = rec

    return recs_flt


class TimeAdjuster:
    """
    Find appropriate date and adjust date.
    """

    def __init__(self, future=True, end_time=None):
        self._future = future
        self.cals = D.calendar(future=future, end_time=end_time)

    def set_end_time(self, end_time=None):
        """
        Set end time. None for use calendar's end time.

        Args:
            end_time
        """
        self.cals = D.calendar(future=self._future, end_time=end_time)

    def get(self, idx: int):
        """
        Get datetime by index.

        Parameters
        ----------
        idx : int
            index of the calendar
        """
        if idx >= len(self.cals):
            return None
        return self.cals[idx]

    def max(self) -> pd.Timestamp:
        """
        Return the max calendar datetime
        """
        return max(self.cals)

    def align_idx(self, time_point, tp_type="start") -> int:
        """
        Align the index of time_point in the calendar.

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

    def cal_interval(self, time_point_A, time_point_B) -> int:
        """
        Calculate the trading day interval (time_point_A - time_point_B)

        Args:
            time_point_A : time_point_A
            time_point_B : time_point_B (is the past of time_point_A)

        Returns:
            int: the interval between A and B
        """
        return self.align_idx(time_point_A) - self.align_idx(time_point_B)

    def align_time(self, time_point, tp_type="start") -> pd.Timestamp:
        """
        Align time_point to trade date of calendar

        Args:
            time_point
                Time point
            tp_type : str
                time point type (`"start"`, `"end"`)

        Returns:
            pd.Timestamp
        """
        return self.cals[self.align_idx(time_point, tp_type=tp_type)]

    def align_seg(self, segment: Union[dict, tuple]) -> Union[dict, tuple]:
        """
        Align the given date to the trade date

        for example:

            .. code-block:: python

                input: {'train': ('2008-01-01', '2014-12-31'), 'valid': ('2015-01-01', '2016-12-31'), 'test': ('2017-01-01', '2020-08-01')}

                output: {'train': (Timestamp('2008-01-02 00:00:00'), Timestamp('2014-12-31 00:00:00')),
                        'valid': (Timestamp('2015-01-05 00:00:00'), Timestamp('2016-12-30 00:00:00')),
                        'test': (Timestamp('2017-01-03 00:00:00'), Timestamp('2020-07-31 00:00:00'))}

        Parameters
        ----------
        segment

        Returns
        -------
        Union[dict, tuple]: the start and end trade date (pd.Timestamp) between the given start and end date.
        """
        if isinstance(segment, dict):
            return {k: self.align_seg(seg) for k, seg in segment.items()}
        elif isinstance(segment, tuple) or isinstance(segment, list):
            return self.align_time(segment[0], tp_type="start"), self.align_time(segment[1], tp_type="end")
        else:
            raise NotImplementedError(f"This type of input is not supported")

    def truncate(self, segment: tuple, test_start, days: int) -> tuple:
        """
        Truncate the segment based on the test_start date

        Parameters
        ----------
        segment : tuple
            time segment
        test_start
        days : int
            The trading days to be truncated
            the data in this segment may need 'days' data

        Returns
        ---------
        tuple: new segment
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

    def shift(self, seg: tuple, step: int, rtype=SHIFT_SD) -> tuple:
        """
        Shift the datatime of segment

        Parameters
        ----------
        seg :
            datetime segment
        step : int
            rolling step
        rtype : str
            rolling type ("sliding" or "expanding")

        Returns
        --------
        tuple: new segment

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
