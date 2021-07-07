# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Base exception class
class QlibException(Exception):
    def __init__(self, message):
        super(QlibException, self).__init__(message)


# Error type for reinitialization when starting an experiment
class RecorderInitializationError(QlibException):
    pass


# Error type for Recorder when can not load object
class LoadObjectError(QlibException):
    pass
