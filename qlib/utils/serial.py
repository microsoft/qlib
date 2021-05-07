# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from pathlib import Path
import pickle
from typing import Union


class Serializable:
    """
    Serializable will change the behaviors of pickle.
    - It only saves the state whose name **does not** start with `_`
    It provides a syntactic sugar for distinguish the attributes which user doesn't want.
    - For examples, a learnable Datahandler just wants to save the parameters without data when dumping to disk
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
        What attribute will not be dumped
        """
        return getattr(self, "_exclude", [])

    FLAG_KEY = "_qlib_serial_flag"

    def config(self, dump_all: bool = None, exclude: list = None, recursive=False):
        """
        configure the serializable object

        Parameters
        ----------
        dump_all : bool
            will the object dump all object
        exclude : list
            What attribute will not be dumped
        recursive : bool
            will the configuration be recursive
        """

        params = {"dump_all": dump_all, "exclude": exclude}

        for k, v in params.items():
            if v is not None:
                attr_name = f"_{k}"
                setattr(self, attr_name, v)

        if recursive:
            for obj in self.__dict__.values():
                # set flag to prevent endless loop
                self.__dict__[self.FLAG_KEY] = True
                if isinstance(obj, Serializable) and self.FLAG_KEY not in obj.__dict__:
                    obj.config(**params, recursive=True)
                del self.__dict__[self.FLAG_KEY]

    def to_pickle(self, path: Union[Path, str], dump_all: bool = None, exclude: list = None):
        self.config(dump_all=dump_all, exclude=exclude)
        with Path(path).open("wb") as f:
            pickle.dump(self, f)
