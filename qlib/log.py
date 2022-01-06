# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import logging
from typing import Optional, Text, Dict, Any
import re
from logging import config as logging_config
from time import time
from contextlib import contextmanager

from .config import C


class MetaLogger(type):
    def __new__(mcs, name, bases, attrs):
        wrapper_dict = logging.Logger.__dict__.copy()
        for key in wrapper_dict:
            if key not in attrs and key != "__reduce__":
                attrs[key] = wrapper_dict[key]
        return type.__new__(mcs, name, bases, attrs)


class QlibLogger(metaclass=MetaLogger):
    """
    Customized logger for Qlib.
    """

    def __init__(self, module_name):
        self.module_name = module_name
        # this feature name conflicts with the attribute with Logger
        # rename it to avoid some corner cases that result in comparing `str` and `int`
        self.__level = 0

    @property
    def logger(self):
        logger = logging.getLogger(self.module_name)
        logger.setLevel(self.__level)
        return logger

    def setLevel(self, level):
        self.__level = level

    def __getattr__(self, name):
        # During unpickling, python will call __getattr__. Use this line to avoid maximum recursion error.
        if name in {"__setstate__"}:
            raise AttributeError
        return self.logger.__getattribute__(name)


def get_module_logger(module_name, level: Optional[int] = None) -> QlibLogger:
    """
    Get a logger for a specific module.

    :param module_name: str
        Logic module name.
    :param level: int
    :return: Logger
        Logger object.
    """
    if level is None:
        level = C.logging_level

    module_name = "qlib.{}".format(module_name)
    # Get logger.
    module_logger = QlibLogger(module_name)
    module_logger.setLevel(level)
    return module_logger


class TimeInspector:

    timer_logger = get_module_logger("timer", level=logging.INFO)

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
            Info that will be logged into stdout.
        """
        cost_time = time() - cls.time_marks.pop()
        cls.timer_logger.info("Time cost: {0:.3f}s | {1}".format(cost_time, info))

    @classmethod
    @contextmanager
    def logt(cls, name="", show_start=False):
        """logt.
        Log the time of the inside code

        Parameters
        ----------
        name :
            name
        show_start :
            show_start
        """
        if show_start:
            cls.timer_logger.info(f"{name} Begin")
        cls.set_time_mark()
        try:
            yield None
        finally:
            pass
        cls.log_cost_time(info=f"{name} Done")


def set_log_with_config(log_config: Dict[Text, Any]):
    """set log with config

    :param log_config:
    :return:
    """
    logging_config.dictConfig(log_config)


class LogFilter(logging.Filter):
    def __init__(self, param=None):
        super().__init__()
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


def set_global_logger_level(level: int, return_orig_handler_level: bool = False):
    """set qlib.xxx logger handlers level

    Parameters
    ----------
    level: int
        logger level

    return_orig_handler_level: bool
        return origin handler level map

    Examples
    ---------

        .. code-block:: python

            import qlib
            import logging
            from qlib.log import get_module_logger, set_global_logger_level
            qlib.init()

            tmp_logger_01 = get_module_logger("tmp_logger_01", level=logging.INFO)
            tmp_logger_01.info("1. tmp_logger_01 info show")

            global_level = logging.WARNING + 1
            set_global_logger_level(global_level)
            tmp_logger_02 = get_module_logger("tmp_logger_02", level=logging.INFO)
            tmp_logger_02.log(msg="2. tmp_logger_02 log show", level=global_level)

            tmp_logger_01.info("3. tmp_logger_01 info do not show")

    """
    _handler_level_map = {}
    qlib_logger = logging.root.manager.loggerDict.get("qlib", None)
    if qlib_logger is not None:
        for _handler in qlib_logger.handlers:
            _handler_level_map[_handler] = _handler.level
            _handler.level = level
    return _handler_level_map if return_orig_handler_level else None


@contextmanager
def set_global_logger_level_cm(level: int):
    """set qlib.xxx logger handlers level to use contextmanager

    Parameters
    ----------
    level: int
        logger level

    Examples
    ---------

        .. code-block:: python

            import qlib
            import logging
            from qlib.log import get_module_logger, set_global_logger_level_cm
            qlib.init()

            tmp_logger_01 = get_module_logger("tmp_logger_01", level=logging.INFO)
            tmp_logger_01.info("1. tmp_logger_01 info show")

            global_level = logging.WARNING + 1
            with set_global_logger_level_cm(global_level):
                tmp_logger_02 = get_module_logger("tmp_logger_02", level=logging.INFO)
                tmp_logger_02.log(msg="2. tmp_logger_02 log show", level=global_level)
                tmp_logger_01.info("3. tmp_logger_01 info do not show")

            tmp_logger_01.info("4. tmp_logger_01 info show")

    """
    _handler_level_map = set_global_logger_level(level, return_orig_handler_level=True)
    try:
        yield
    finally:
        for _handler, _level in _handler_level_map.items():
            _handler.level = _level
