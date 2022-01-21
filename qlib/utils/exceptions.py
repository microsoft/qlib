# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


# Base exception class
class QlibException(Exception):
    pass


class RecorderInitializationError(QlibException):
    """Error type for re-initialization when starting an experiment"""


class LoadObjectError(QlibException):
    """Error type for Recorder when can not load object"""


class ExpAlreadyExistError(Exception):
    """Experiment already exists"""
