from typing import Callable
import pandas as pd
from qlib.config import C
from qlib.data import D
from qlib import get_module_logger
from qlib.log import set_log_with_config
from qlib.model.ens.ensemble import ens_workflow
from qlib.workflow.online.manager import OnlineManager
from qlib.workflow.task.collect import Collector


class OnlineSimulator:
    """
    To simulate online serving in the past, like a "online serving backtest".
    """

    def __init__(
        self,
        start_time,
        end_time,
        onlinemanager: OnlineManager,
        frequency="day",
        time_delta="20 hours",
        collector: Collector = None,
        process_list: list = None,
    ):
        self.logger = get_module_logger(self.__class__.__name__)
        self.cal = D.calendar(start_time=start_time, end_time=end_time, freq=frequency)
        self.start_time = self.cal[0]
        self.end_time = self.cal[-1]
        self.olm = onlinemanager
        self.time_delta = time_delta

        if len(self.cal) == 0:
            self.logger.warn(f"There is no need to simulate bacause start_time is larger than end_time.")
        self.collector = collector
        self.process_list = process_list

    def simulate(self, *args, **kwargs):
        """
        Starting from start time, this method will simulate every routine in OnlineManager.
        NOTE: Considering the parallel training, the signals will be perpared after all routine simulating.

        Returns:
            dict: the simulated results collected by collector
        """
        self.rec_dict = {}
        tmp_begin = self.start_time
        tmp_end = None
        prev_recorders = self.olm.online_models()
        for cur_time in self.cal:
            cur_time = cur_time + pd.Timedelta(self.time_delta)
            self.logger.info(f"Simulating at {str(cur_time)}......")
            recorders = self.olm.routine(cur_time, True, *args, **kwargs)
            if len(recorders) == 0:
                tmp_end = cur_time
            else:
                self.rec_dict[(tmp_begin, tmp_end)] = prev_recorders
                tmp_begin = cur_time
                prev_recorders = recorders

        self.rec_dict[(tmp_begin, self.end_time)] = prev_recorders
        # prepare signals again incase there is no trained model when call it
        self.olm.run_delay_signals()
        self.logger.info(f"Finished preparing signals")

        if self.collector is not None:
            return ens_workflow(self.collector, self.process_list)

    def online_models(self):
        """
        Return a online models dict likes {(begin_time, end_time):[online models]}.

        Returns:
            dict
        """
        if hasattr(self, "rec_dict"):
            return self.rec_dict
        self.logger.warn(f"Please call `simulate` firstly when calling `online_models`")
        return {}
