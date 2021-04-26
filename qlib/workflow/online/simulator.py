from qlib.data import D
from qlib import get_module_logger
from qlib.workflow.online.manager import OnlineManager


class OnlineSimulator:
    """
    To simulate online serving in the past, like a "online serving backtest".
    """

    def __init__(
        self,
        start_time,
        end_time,
        online_manager: OnlineManager,
        frequency="day",
    ):
        """
        init OnlineSimulator.

        Args:
            start_time (str or pd.Timestamp): the start time of simulating.
            end_time (str or pd.Timestamp): the end time of simulating. If None, then end_time is latest.
            onlinemanager (OnlineManager): the instance of OnlineManager
            frequency (str, optional): the data frequency. Defaults to "day".
        """
        self.logger = get_module_logger(self.__class__.__name__)
        self.cal = D.calendar(start_time=start_time, end_time=end_time, freq=frequency)
        self.start_time = self.cal[0]
        self.end_time = self.cal[-1]
        self.olm = online_manager
        if len(self.cal) == 0:
            self.logger.warn(f"There is no need to simulate bacause start_time is larger than end_time.")

    def simulate(self, *args, **kwargs):
        """
        Starting from start time, this method will simulate every routine in OnlineManager.
        NOTE: Considering the parallel training, the models and signals can be perpared after all routine simulating.

        Returns:
            Collector: the OnlineManager's collector
        """
        self.rec_dict = {}
        tmp_begin = self.start_time
        tmp_end = None
        prev_recorders = self.olm.online_models()
        for cur_time in self.cal:
            self.logger.info(f"Simulating at {str(cur_time)}......")
            recorders = self.olm.routine(cur_time, True, *args, **kwargs)
            if len(recorders) == 0:
                tmp_end = cur_time
            else:
                self.rec_dict[(tmp_begin, tmp_end)] = prev_recorders
                tmp_begin = cur_time
                prev_recorders = recorders
        self.rec_dict[(tmp_begin, self.end_time)] = prev_recorders
        # finished perparing models (and pred) and signals
        self.olm.delay_prepare(self.rec_dict)
        self.logger.info(f"Finished preparing signals")
        return self.olm.get_collector()

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
