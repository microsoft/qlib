"""
This module will base on Qlib's logger module and provides some interactive functions.
"""
import logging

from typing import Dict, List
from qlib.finco.utils import Singleton
from qlib.log import get_module_logger
from contextlib import contextmanager


class LogColors:
    """
    ANSI color codes for use in console output.
    """
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'
    WHITE = '\033[97m'
    GRAY = '\033[90m'
    BLACK = '\033[30m'


@contextmanager
def formatting_log(logger, title="Info"):
    """
    a context manager, print liens before and after a function
    """
    length = {"Task": 120, "Info": 0}.get(title, 60)
    start = int((length - len(title)) / 2)
    pad = 1 if len(title) % 2 else 0
    color, bold = (LogColors.YELLOW, LogColors.BOLD) if title in ["Info", "Task"] else (LogColors.CYAN, '')
    logger.info("")
    logger.info(f"{color}{bold}{'-' * (start + pad)} {title} {'-' * start}{LogColors.END}")
    yield
    logger.info("")


class FinCoLog(Singleton):
    # TODO:
    # - config to file logger and save it into workspace
    def __init__(self) -> None:
        # self.logger = get_module_logger("interactive")
        self.logger = logging.Logger("interactive")

        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def log_message(self, messages: List[Dict[str, str]]):
        """
        messages is some info like this  [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]
        """
        with formatting_log(self.logger, "GPT Messages"):
            for m in messages:
                self.logger.info(
                    f"\n{LogColors.MAGENTA}{LogColors.BOLD}Role:{LogColors.END} "
                    f"{LogColors.CYAN}{m['role']}{LogColors.END}\n"
                    + f"{LogColors.MAGENTA}{LogColors.BOLD}Content:{LogColors.END} "
                      f"{LogColors.CYAN}{m['content']}{LogColors.END}")

    def log_response(self, response: str):
        with formatting_log(self.logger, "GPT Response"):
            self.logger.info(
                f"\n{LogColors.CYAN}{response}{LogColors.END}\n")

    # TODO:
    # It looks wierd if we only have logger
    def info(self, *args, plain=False, title="Info"):
        if plain:
            return self.plain_info(*args)
        with formatting_log(self.logger, title):
            for arg in args:
                self.logger.info(f"{LogColors.WHITE}{arg}{LogColors.END}")

    def plain_info(self, *args):
        for arg in args:
            # self.logger.info(arg)
            self.logger.info(
                f"{LogColors.YELLOW}{LogColors.BOLD}Info:{LogColors.END}{LogColors.WHITE}{arg}{LogColors.END}")
