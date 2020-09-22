# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import os
import re
import logging
from time import time
import logging.handlers
from logging import config as logging_config

from .config import C


def get_module_logger(module_name, level=None):
    """
    Get a logger for a specific module.

    :param module_name: str
        Logic module name.
    :param level: int
    :param sh_level: int
        Stream handler log level.
    :param log_format: str
    :return: Logger
        Logger object.
    """
    if level is None:
        level = C.logging_level

    module_name = "qlib.{}".format(module_name)
    # Get logger.
    module_logger = logging.getLogger(module_name)
    module_logger.setLevel(level)
    return module_logger


class TimeInspector(object):

    timer_logger = get_module_logger("timer", level=logging.WARNING)

    time_marks = []

    @classmethod
    def set_time_mark(cls):
        """
        Set a time mark with current time, and this time mark will push into a stack.
        :return: float
            A timestamp for current time.
        """
        _time = time()
        cls.time_marks.append(_time)
        return _time

    @classmethod
    def pop_time_mark(cls):
        """
        Pop last time mark from stack.
        """
        return cls.time_marks.pop()

    @classmethod
    def get_cost_time(cls):
        """
        Get last time mark from stack, calculate time diff with current time.
        :return: float
            Time diff calculated by last time mark with current time.
        """
        cost_time = time() - cls.time_marks.pop()
        return cost_time

    @classmethod
    def log_cost_time(cls, info="Done"):
        """
        Get last time mark from stack, calculate time diff with current time, and log time diff and info.
        :param info: str
            Info that will be log into stdout.
        """
        cost_time = time() - cls.time_marks.pop()
        cls.timer_logger.info("Time cost: {0:.5f} | {1}".format(cost_time, info))


def set_log_with_config(log_config: dict):
    """set log with config

    :param log_config:
    :return:
    """
    logging_config.dictConfig(log_config)


class LogFilter(logging.Filter):
    def __init__(self, param=None):
        self.param = param

    @staticmethod
    def match_msg(filter_str, msg):
        match = False
        try:
            if re.match(filter_str, msg):
                match = True
        except Exception:
            pass
        return match

    def filter(self, record):
        allow = True
        if isinstance(self.param, str):
            allow = not self.match_msg(self.param, record.msg)
        elif isinstance(self.param, list):
            allow = not any([self.match_msg(p, record.msg) for p in self.param])
        return allow
