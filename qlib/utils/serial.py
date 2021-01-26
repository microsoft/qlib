# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from pathlib import Path
import pickle


class Serializable:
    """
    Serializable behaves like pickle.
    But it only saves the state whose name **does not** start with `_`
    """

    def __init__(self):
        self._dump_all = False
        self._exclude = []

    def __getstate__(self) -> dict:
        return {
            k: v for k, v in self.__dict__.items() if k not in self.exclude and (self.dump_all or not k.startswith("_"))
        }

    def __setstate__(self, state: dict):
        self.__dict__.update(state)

    @property
    def dump_all(self):
        """
        will the object dump all object
        """
        return getattr(self, "_dump_all", False)

    @property
    def exclude(self):
        """
        What attribute will be dumped
        """
        return getattr(self, "_exclude", [])

    def config(self, dump_all: bool = None, exclude: list = None):
        if dump_all is not None:
            self._dump_all = dump_all

        if exclude is not None:
            self._exclude = exclude

    def to_pickle(self, path: [Path, str], dump_all: bool = None, exclude: list = None):
        self.config(dump_all=dump_all, exclude=exclude)
        with Path(path).open("wb") as f:
            pickle.dump(self, f)
