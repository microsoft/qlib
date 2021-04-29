# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
OnlineManager can manage a set of OnlineStrategy and run them dynamically.

With the change of time, the decisive models will be also changed. In this module, we call those contributing models as `online` models.
In every routine(such as everyday or every minutes), the `online` models maybe changed and the prediction of them need to be updated.
So this module provide a series methods to control this process. 
"""

from typing import Dict, List, Union

import pandas as pd
from qlib import get_module_logger
from qlib.data.data import D
from qlib.utils.serial import Serializable
from qlib.workflow.online.strategy import OnlineStrategy
from qlib.workflow.task.collect import HyperCollector


class OnlineManager(Serializable):
    def __init__(
        self,
        strategy: Union[OnlineStrategy, List[OnlineStrategy]],
        begin_time: Union[str, pd.Timestamp] = None,
        freq="day",
        need_log=True,
    ):
        """
        Init OnlineManager.

        Args:
            strategy (Union[OnlineStrategy, List[OnlineStrategy]]): an instance of OnlineStrategy or a list of OnlineStrategy
            begin_time (Union[str,pd.Timestamp], optional): the OnlineManager will begin at this time. Defaults to None.
            freq (str, optional): data frequency. Defaults to "day".
            need_log (bool, optional): print log or not. Defaults to True.
        """
        self.logger = get_module_logger(self.__class__.__name__)
        self.need_log = need_log
        if not isinstance(strategy, list):
            strategy = [strategy]
        self.strategy = strategy
        self.freq = freq
        if begin_time is None:
            begin_time = D.calendar(freq=self.freq).max()
        self.begin_time = pd.Timestamp(begin_time)
        self.cur_time = self.begin_time
        self.history = {}

    def first_train(self):
        """
        Run every strategy first_train method and record the online history
        """
        for strategy in self.strategy:
            self.logger.info(f"Strategy `{strategy.name_id}` begins first training...")
            online_models = strategy.first_train()
            self.history.setdefault(strategy.name_id, {})[self.cur_time] = online_models

    def routine(self, cur_time: Union[str, pd.Timestamp] = None, task_kwargs: dict = {}, model_kwargs: dict = {}):
        """
        Run typical update process for every strategy and record the online history.

        The typical update process after a routine, such as day by day or month by month.
        update online prediction -> prepare signals -> prepare tasks -> prepare new models -> reset online models

        Args:
            cur_time (Union[str,pd.Timestamp], optional): run routine method in this time. Defaults to None.
            task_kwargs (dict): the params for `prepare_tasks`
            model_kwargs (dict): the params for `prepare_online_models`
        """
        if cur_time is None:
            cur_time = D.calendar(freq=self.freq).max()
        self.cur_time = pd.Timestamp(cur_time)  # None for latest date
        for strategy in self.strategy:
            if self.need_log:
                self.logger.info(f"Strategy `{strategy.name_id}` begins routine...")
            if not strategy.trainer.is_delay():
                strategy.prepare_signals()
            tasks = strategy.prepare_tasks(self.cur_time, **task_kwargs)
            online_models = strategy.prepare_online_models(tasks, **model_kwargs)
            if len(online_models) > 0:
                self.history.setdefault(strategy.name_id, {})[self.cur_time] = online_models

    def get_collector(self) -> HyperCollector:
        """
        Get the instance of HyperCollector to collect results from every strategy.

        Returns:
            HyperCollector: the collector can collect other collectors.
        """
        collector_dict = {}
        for strategy in self.strategy:
            collector_dict[strategy.name_id] = strategy.get_collector()
        return HyperCollector(collector_dict)

    def get_online_history(self, strategy_name_id: str) -> list:
        """
        Get the online history based on strategy_name_id.

        Args:
            strategy_name_id (str): the name_id of strategy

        Returns:
            dict: a list like [(time, [online_models])]
        """
        history_dict = self.history[strategy_name_id]
        history = []
        for time in sorted(history_dict):
            models = history_dict[time]
            history.append((time, models))
        return history

    def delay_prepare(self, delay_kwargs={}):
        """
        Prepare all models and signals if there are something waiting for prepare.

        Args:
            delay_kwargs: the params for `delay_prepare`
        """
        for strategy in self.strategy:
            strategy.delay_prepare(self.get_online_history(strategy.name_id), **delay_kwargs)

    def simulate(self, end_time, frequency="day", task_kwargs={}, model_kwargs={}, delay_kwargs={}) -> HyperCollector:
        """
        Starting from cur time, this method will simulate every routine in OnlineManager.
        NOTE: Considering the parallel training, the models and signals can be perpared after all routine simulating.

        Returns:
            HyperCollector: the OnlineManager's collector
        """
        cal = D.calendar(start_time=self.cur_time, end_time=end_time, freq=frequency)
        self.first_train()
        for cur_time in cal:
            self.logger.info(f"Simulating at {str(cur_time)}......")
            self.routine(cur_time, task_kwargs=task_kwargs, model_kwargs=model_kwargs)
        self.delay_prepare(delay_kwargs=delay_kwargs)
        self.logger.info(f"Finished preparing signals")
        return self.get_collector()

    def reset(self):
        """
        NOTE: This method will reset all strategy! Be careful to use it.
        """
        self.cur_time = self.begin_time
        self.history = {}
        for strategy in self.strategy:
            strategy.reset()
