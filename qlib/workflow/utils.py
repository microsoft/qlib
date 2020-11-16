# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys, traceback, signal
from . import R
from .recorder import Recorder
from ..log import get_module_logger

logger = get_module_logger("workflow", "INFO")


def experiment_exception_hook(type, value, tb):
    """
    End an experiment with status to be "FAILED". This exception tries to catch those uncaught exception
    and end the experiment automatically.

    Parameters
    type: Exception type
    value: Exception's value
    tb: Exception's traceback
    """
    error_msg = "An exception has been raised.\n" f"Type: {type}\n" f"Value: {value}\n"
    logger.error(error_msg)
    traceback.print_tb(tb)

    R.end_exp(recorder_status=Recorder.STATUS_FA)


def experiment_kill_signal_handler(signum, frame):
    """
    End an experiment when user kill the program (CTRL+C, etc.).
    """
    R.end_exp(recorder_status=Recorder.STATUS_FA)
