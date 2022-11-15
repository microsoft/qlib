# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
OnlineManager can manage a set of `Online Strategy <#Online Strategy>`_ and run them dynamically.

With the change of time, the decisive models will be also changed. In this module, we call those contributing models `online` models.
In every routine(such as every day or every minute), the `online` models may be changed and the prediction of them needs to be updated.
So this module provides a series of methods to control this process.

This module also provides a method to simulate `Online Strategy <#Online Strategy>`_ in history.
Which means you can verify your strategy or find a better one.

There are 4 total situations for using different trainers in different situations:



=========================  ===================================================================================
Situations                 Description
=========================  ===================================================================================
Online + Trainer           When you want to do a REAL routine, the Trainer will help you train the models. It
                           will train models task by task and strategy by strategy.

Online + DelayTrainer      DelayTrainer will skip concrete training until all tasks have been prepared by
                           different strategies. It makes users can parallelly train all tasks at the end of
                           `routine` or `first_train`. Otherwise, these functions will get stuck when each
                           strategy prepare tasks.

Simulation + Trainer       It will behave in the same way as `Online + Trainer`. The only difference is that it
                           is for simulation/backtesting instead of online trading

Simulation + DelayTrainer  When your models don't have any temporal dependence, you can use DelayTrainer
                           for the ability to multitasking. It means all tasks in all routines
                           can be REAL trained at the end of simulating. The signals will be prepared well at
                           different time segments (based on whether or not any new model is online).
=========================  ===================================================================================

Here is some pseudo code the demonstrate the workflow of each situation

For simplicity
    - Only one strategy is used in the strategy
    - `update_online_pred` is only called in the online mode and is ignored

1) `Online + Trainer`

.. code-block:: python

    tasks = first_train()
    models = trainer.train(tasks)
    trainer.end_train(models)
    for day in online_trading_days:
        # OnlineManager.routine
        models = trainer.train(strategy.prepare_tasks())  # for each strategy
        strategy.prepare_online_models(models)  # for each strategy

        trainer.end_train(models)
        prepare_signals()  # prepare trading signals daily


`Online + DelayTrainer`: the workflow is the same as `Online + Trainer`.


2) `Simulation + DelayTrainer`

.. code-block:: python

    # simulate
    tasks = first_train()
    models = trainer.train(tasks)
    for day in historical_calendars:
        # OnlineManager.routine
        models = trainer.train(strategy.prepare_tasks())  # for each strategy
        strategy.prepare_online_models(models)  # for each strategy
    # delay_prepare()
    # FIXME: Currently the delay_prepare is not implemented in a proper way.
    trainer.end_train(<for all previous models>)
    prepare_signals()


# Can we simplify current workflow?

- Can reduce the number of state of tasks?

    - For each task, we have three phases (i.e. task, partly trained task, final trained task)
"""

import logging
from typing import Callable, List, Union

import pandas as pd
from qlib import get_module_logger
from qlib.data.data import D
from qlib.log import set_global_logger_level
from qlib.model.ens.ensemble import AverageEnsemble
from qlib.model.trainer import Trainer, TrainerR
from qlib.utils.serial import Serializable
from qlib.workflow.online.strategy import OnlineStrategy
from qlib.workflow.task.collect import MergeCollector


class OnlineManager(Serializable):
    """
    OnlineManager can manage online models with `Online Strategy <#Online Strategy>`_.
    It also provides a history recording of which models are online at what time.
    """

    STATUS_SIMULATING = "simulating"  # when calling `simulate`
    STATUS_ONLINE = "online"  # the normal status. It is used when online trading

    def __init__(
        self,
        strategies: Union[OnlineStrategy, List[OnlineStrategy]],
        trainer: Trainer = None,
        begin_time: Union[str, pd.Timestamp] = None,
        freq="day",
    ):
        """
        Init OnlineManager.
        One OnlineManager must have at least one OnlineStrategy.

        Args:
            strategies (Union[OnlineStrategy, List[OnlineStrategy]]): an instance of OnlineStrategy or a list of OnlineStrategy
            begin_time (Union[str,pd.Timestamp], optional): the OnlineManager will begin at this time. Defaults to None for using the latest date.
            trainer (Trainer): the trainer to train task. None for using TrainerR.
            freq (str, optional): data frequency. Defaults to "day".
        """
        self.logger = get_module_logger(self.__class__.__name__)
        if not isinstance(strategies, list):
            strategies = [strategies]
        self.strategies = strategies
        self.freq = freq
        if begin_time is None:
            begin_time = D.calendar(freq=self.freq).max()
        self.begin_time = pd.Timestamp(begin_time)
        self.cur_time = self.begin_time
        # OnlineManager will recorder the history of online models, which is a dict like {pd.Timestamp, {strategy, [online_models]}}.
        # It records the online servnig models of each strategy for each day.
        self.history = {}
        if trainer is None:
            trainer = TrainerR()
        self.trainer = trainer
        self.signals = None
        self.status = self.STATUS_ONLINE

    def _postpone_action(self):
        """
        Should the workflow to postpone the following actions to the end (in delay_prepare)
        - trainer.end_train
        - prepare_signals

        Postpone these actions is to support simulating/backtest online strategies without time dependencies.
        All the actions can be done parallelly at the end.
        """
        return self.status == self.STATUS_SIMULATING and self.trainer.is_delay()

    def first_train(self, strategies: List[OnlineStrategy] = None, model_kwargs: dict = {}):
        """
        Get tasks from every strategy's first_tasks method and train them.
        If using DelayTrainer, it can finish training all together after every strategy's first_tasks.

        Args:
            strategies (List[OnlineStrategy]): the strategies list (need this param when adding strategies). None for use default strategies.
            model_kwargs (dict): the params for `prepare_online_models`
        """
        if strategies is None:
            strategies = self.strategies

        models_list = []
        for strategy in strategies:
            self.logger.info(f"Strategy `{strategy.name_id}` begins first training...")
            tasks = strategy.first_tasks()
            models = self.trainer.train(tasks, experiment_name=strategy.name_id)
            models_list.append(models)
            self.logger.info(f"Finished training {len(models)} models.")
            # FIXME: Train multiple online models at `first_train` will result in getting too much online models at the
            # start.
            online_models = strategy.prepare_online_models(models, **model_kwargs)
            self.history.setdefault(self.cur_time, {})[strategy] = online_models

        if not self._postpone_action():
            for strategy, models in zip(strategies, models_list):
                models = self.trainer.end_train(models, experiment_name=strategy.name_id)

    def routine(
        self,
        cur_time: Union[str, pd.Timestamp] = None,
        task_kwargs: dict = {},
        model_kwargs: dict = {},
        signal_kwargs: dict = {},
    ):
        """
        Typical update process for every strategy and record the online history.

        The typical update process after a routine, such as day by day or month by month.
        The process is: Update predictions -> Prepare tasks -> Prepare online models -> Prepare signals.

        If using DelayTrainer, it can finish training all together after every strategy's prepare_tasks.

        Args:
            cur_time (Union[str,pd.Timestamp], optional): run routine method in this time. Defaults to None.
            task_kwargs (dict): the params for `prepare_tasks`
            model_kwargs (dict): the params for `prepare_online_models`
            signal_kwargs (dict): the params for `prepare_signals`
        """
        if cur_time is None:
            cur_time = D.calendar(freq=self.freq).max()
        self.cur_time = pd.Timestamp(cur_time)  # None for latest date

        models_list = []
        for strategy in self.strategies:
            self.logger.info(f"Strategy `{strategy.name_id}` begins routine...")

            tasks = strategy.prepare_tasks(self.cur_time, **task_kwargs)
            models = self.trainer.train(tasks, experiment_name=strategy.name_id)
            models_list.append(models)
            self.logger.info(f"Finished training {len(models)} models.")
            online_models = strategy.prepare_online_models(models, **model_kwargs)
            self.history.setdefault(self.cur_time, {})[strategy] = online_models

            # The online model may changes in the above processes
            # So updating the predictions of online models should be the last step
            if self.status == self.STATUS_ONLINE:
                strategy.tool.update_online_pred()

        if not self._postpone_action():
            for strategy, models in zip(self.strategies, models_list):
                models = self.trainer.end_train(models, experiment_name=strategy.name_id)
            self.prepare_signals(**signal_kwargs)

    def get_collector(self, **kwargs) -> MergeCollector:
        """
        Get the instance of `Collector <../advanced/task_management.html#Task Collecting>`_ to collect results from every strategy.
        This collector can be a basis as the signals preparation.

        Args:
            **kwargs: the params for get_collector.

        Returns:
            MergeCollector: the collector to merge other collectors.
        """
        collector_dict = {}
        for strategy in self.strategies:
            collector_dict[strategy.name_id] = strategy.get_collector(**kwargs)
        return MergeCollector(collector_dict, process_list=[])

    def add_strategy(self, strategies: Union[OnlineStrategy, List[OnlineStrategy]]):
        """
        Add some new strategies to OnlineManager.

        Args:
            strategy (Union[OnlineStrategy, List[OnlineStrategy]]): a list of OnlineStrategy
        """
        if not isinstance(strategies, list):
            strategies = [strategies]
        self.first_train(strategies)
        self.strategies.extend(strategies)

    def prepare_signals(self, prepare_func: Callable = AverageEnsemble(), over_write=False):
        """
        After preparing the data of the last routine (a box in box-plot) which means the end of the routine, we can prepare trading signals for the next routine.

        NOTE: Given a set prediction, all signals before these prediction end times will be prepared well.

        Even if the latest signal already exists, the latest calculation result will be overwritten.

        .. note::

            Given a prediction of a certain time, all signals before this time will be prepared well.

        Args:
            prepare_func (Callable, optional): Get signals from a dict after collecting. Defaults to AverageEnsemble(), the results collected by MergeCollector must be {xxx:pred}.
            over_write (bool, optional): If True, the new signals will overwrite. If False, the new signals will append to the end of signals. Defaults to False.

        Returns:
            pd.DataFrame: the signals.
        """
        signals = prepare_func(self.get_collector()())
        old_signals = self.signals
        if old_signals is not None and not over_write:
            old_max = old_signals.index.get_level_values("datetime").max()
            new_signals = signals.loc[old_max:]
            signals = pd.concat([old_signals, new_signals], axis=0)
        else:
            new_signals = signals
        self.logger.info(f"Finished preparing new {len(new_signals)} signals.")
        self.signals = signals
        return new_signals

    def get_signals(self) -> Union[pd.Series, pd.DataFrame]:
        """
        Get prepared online signals.

        Returns:
            Union[pd.Series, pd.DataFrame]: pd.Series for only one signals every datetime.
            pd.DataFrame for multiple signals, for example, buy and sell operations use different trading signals.
        """
        return self.signals

    SIM_LOG_LEVEL = logging.INFO + 1  # when simulating, reduce information
    SIM_LOG_NAME = "SIMULATE_INFO"

    def simulate(
        self, end_time=None, frequency="day", task_kwargs={}, model_kwargs={}, signal_kwargs={}
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Starting from the current time, this method will simulate every routine in OnlineManager until the end time.

        Considering the parallel training, the models and signals can be prepared after all routine simulating.

        The delay training way can be ``DelayTrainer`` and the delay preparing signals way can be ``delay_prepare``.

        Args:
            end_time: the time the simulation will end
            frequency: the calendar frequency
            task_kwargs (dict): the params for `prepare_tasks`
            model_kwargs (dict): the params for `prepare_online_models`
            signal_kwargs (dict): the params for `prepare_signals`

        Returns:
            Union[pd.Series, pd.DataFrame]: pd.Series for only one signals every datetime.
            pd.DataFrame for multiple signals, for example, buy and sell operations use different trading signals.
        """
        self.status = self.STATUS_SIMULATING
        cal = D.calendar(start_time=self.cur_time, end_time=end_time, freq=frequency)
        self.first_train()

        simulate_level = self.SIM_LOG_LEVEL
        set_global_logger_level(simulate_level)
        logging.addLevelName(simulate_level, self.SIM_LOG_NAME)

        for cur_time in cal:
            self.logger.log(level=simulate_level, msg=f"Simulating at {str(cur_time)}......")
            self.routine(
                cur_time,
                task_kwargs=task_kwargs,
                model_kwargs=model_kwargs,
                signal_kwargs=signal_kwargs,
            )
        # delay prepare the models and signals
        if self._postpone_action():
            self.delay_prepare(model_kwargs=model_kwargs, signal_kwargs=signal_kwargs)

        # FIXME: get logging level firstly and restore it here
        set_global_logger_level(logging.DEBUG)
        self.logger.info(f"Finished preparing signals")
        self.status = self.STATUS_ONLINE
        return self.get_signals()

    def delay_prepare(self, model_kwargs={}, signal_kwargs={}):
        """
        Prepare all models and signals if something is waiting for preparation.

        Args:
            model_kwargs: the params for `end_train`
            signal_kwargs: the params for `prepare_signals`
        """
        # FIXME:
        # This method is not implemented in the proper way!!!
        last_models = {}
        signals_time = D.calendar()[0]
        need_prepare = False
        for cur_time, strategy_models in self.history.items():
            self.cur_time = cur_time

            for strategy, models in strategy_models.items():
                # only new online models need to prepare
                if last_models.setdefault(strategy, set()) != set(models):
                    models = self.trainer.end_train(models, experiment_name=strategy.name_id, **model_kwargs)
                    strategy.tool.reset_online_tag(models)
                    need_prepare = True
                last_models[strategy] = set(models)

            if need_prepare:
                # NOTE: Assumption: the predictions of online models need less than next cur_time, or this method will work in a wrong way.
                self.prepare_signals(**signal_kwargs)
                if signals_time > cur_time:
                    # FIXME: if use DelayTrainer and worker (and worker is faster than main progress), there are some possibilities of showing this warning.
                    self.logger.warn(
                        f"The signals have already parpred to {signals_time} by last preparation, but current time is only {cur_time}. This may be because the online models predict more than they should, which can cause signals to be contaminated by the offline models."
                    )
                need_prepare = False
                signals_time = self.signals.index.get_level_values("datetime").max()
