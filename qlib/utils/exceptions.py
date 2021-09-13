# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


# Base exception class
class QlibException(Exception):
    def __init__(self, message):
        super(QlibException, self).__init__(message)


class RecorderInitializationError(QlibException):
    """Error type for re-initialization when starting an experiment"""

    pass


class LoadObjectError(QlibException):
    """Error type for Recorder when can not load object"""

    pass
