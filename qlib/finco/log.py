"""
This module will base on Qlib's logger module and provides some interactive functions.
"""
import logging

from typing import Dict, List
from qlib.finco.utils import SingletonBaseClass
from contextlib import contextmanager


class LogColors:
    """
    ANSI color codes for use in console output.
    """

    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    GRAY = "\033[90m"
    BLACK = "\033[30m"

    BOLD = "\033[1m"
    ITALIC = "\033[3m"

    END = "\033[0m"

    @classmethod
    def get_all_colors(cls):
        names = dir(cls)
        names = [name for name in names if not name.startswith("__") and not callable(getattr(cls, name))]
        var_values = [getattr(cls, name) for name in names]
        return var_values

    def render(self, text: str, color: str = "", style: str = ""):
        """
        render text by input color and style. It's not recommend that input text is already rendered.
        """
        # This method is called too frequently, which is not good.
        colors = self.get_all_colors()
        # Perhaps color and font should be distinguished here.
        if color:
            assert color in colors, f"color should be in: {colors} but now is: {color}"
        if style:
            assert style in colors, f"style should be in: {colors} but now is: {style}"

        text = f"{color}{text}{self.END}"
        text = f"{style}{text}{self.END}"

        return text


@contextmanager
def formatting_log(logger, title="Info"):
    """
    a context manager, print liens before and after a function
    """
    length = {"Start": 120, "Task": 120, "Info": 60, "Interact": 60, "End": 120}.get(title, 60)
    color, bold = (
        (LogColors.YELLOW, LogColors.BOLD)
        if title in ["Start", "Task", "Info", "Interact", "End"]
        else (LogColors.CYAN, "")
    )
    logger.info("")
    logger.info(f"{color}{bold}{'-'} {title} {'-' * (length - len(title))}{LogColors.END}")
    yield
    logger.info("")


class FinCoLog(SingletonBaseClass):
    # TODO:
    # - config to file logger and save it into workspace
    def __init__(self) -> None:
        self.logger = logging.Logger("interactive")
        # TODO:  merge these with Qlib's default logger.
        #  We can do the same thing by changing the default log dict of Qlib.
        #  Reference: https://github.com/microsoft/qlib/blob/main/qlib/config.py#L155

        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
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
                    f"{LogColors.MAGENTA}{LogColors.BOLD}Role:{LogColors.END} "
                    f"{LogColors.CYAN}{m['role']}{LogColors.END}\n"
                    + f"{LogColors.MAGENTA}{LogColors.BOLD}Content:{LogColors.END} "
                    f"{LogColors.CYAN}{m['content']}{LogColors.END}\n"
                )

    def log_response(self, response: str):
        with formatting_log(self.logger, "GPT Response"):
            self.logger.info(f"{LogColors.CYAN}{response}{LogColors.END}\n")

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
            self.logger.info(
                f"{LogColors.YELLOW}{LogColors.BOLD}Info:{LogColors.END}{LogColors.WHITE}{arg}{LogColors.END}"
            )

    def warning(self, *args):
        for arg in args:
            self.logger.warning(f"{LogColors.BLUE}{LogColors.BOLD}Warning:{LogColors.END}{arg}")

    def error(self, *args):
        for arg in args:
            self.logger.error(f"{LogColors.RED}{LogColors.BOLD}Error:{LogColors.END}{arg}")
