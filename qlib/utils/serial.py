# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from pathlib import Path
import pickle
import typing
import dill
from typing import Union


class Serializable:
    """
    Serializable will change the behaviors of pickle.
    - It only saves the state whose name **does not** start with `_`
    It provides a syntactic sugar for distinguish the attributes which user doesn't want.
    - For examples, a learnable Datahandler just wants to save the parameters without data when dumping to disk
    """

    pickle_backend = "pickle"  # another optional value is "dill" which can pickle more things of python.
    default_dump_all = False  # if dump all things

    def __init__(self):
        self._dump_all = self.default_dump_all
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
        """
        Dump self to a pickle file.

        Args:
            path (Union[Path, str]): the path to dump
            dump_all (bool, optional): if need to dump all things. Defaults to None.
            exclude (list, optional): will exclude the attributes in this list when dumping. Defaults to None.
        """
        self.config(dump_all=dump_all, exclude=exclude)
        with Path(path).open("wb") as f:
            self.get_backend().dump(self, f)

    @classmethod
    def load(cls, filepath):
        """
        Load the serializable class from a filepath.

        Args:
            filepath (str): the path of file

        Raises:
            TypeError: the pickled file must be `type(cls)`

        Returns:
            `type(cls)`: the instance of `type(cls)`
        """
        with open(filepath, "rb") as f:
            object = cls.get_backend().load(f)
        if isinstance(object, cls):
            return object
        else:
            raise TypeError(f"The instance of {type(object)} is not a valid `{type(cls)}`!")

    @classmethod
    def get_backend(cls):
        """
        Return the real backend of a Serializable class. The pickle_backend value can be "pickle" or "dill".

        Returns:
            module: pickle or dill module based on pickle_backend
        """
        if cls.pickle_backend == "pickle":
            return pickle
        elif cls.pickle_backend == "dill":
            return dill
        else:
            raise ValueError("Unknown pickle backend, please use 'pickle' or 'dill'.")
