"""
This module will base on Qlib's logger module and provides some interactive functions.
"""
from email import message_from_binary_file
from typing import Dict, List
from qlib.finco.utils import Singleton
from qlib.log import get_module_logger
from contextlib import contextmanager

# a context manager, print liens before and after a function
@contextmanager
def formating_log(logger, text="Interaction"):
    logger.info("")
    logger.info("=" * 20 + f" BEGIN:{text} " + "=" * 20)
    yield
    logger.info("=" * 20 + f" END:  {text} " + "=" * 20)
    logger.info("")


class FinCoLog(Singleton):
    # TODO:
    # - config to file logger and save it into workspace
    def __init__(self) -> None:
        self.logger = get_module_logger("interactive")

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
        with formating_log(self.logger):
            for m in messages:
                self.logger.info(f"Role: {m['role']}")
                self.logger.info(f"Content: {m['content']}")

    # TODO:
    # It looks wierd if we only have logger
    def info(self, *args, **kwargs):
        with formating_log(self.logger, "info"):
            self.logger.info(*args, **kwargs)
