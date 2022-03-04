# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pickle
import dill
from pathlib import Path
from typing import Union
from ..config import C


class Serializable:
    """
    Serializable will change the behaviors of pickle.

        The rule to tell if a attribute will be kept or dropped when dumping.
        The rule with higher priorities is on the top
        - in the config attribute list -> always dropped
        - in the include attribute list -> always kept
        - in the exclude attribute list -> always dropped
        - name not starts with `_` -> kept
        - name starts with `_` -> kept if `dump_all` is true else dropped

    It provides a syntactic sugar for distinguish the attributes which user doesn't want.
    - For examples, a learnable Datahandler just wants to save the parameters without data when dumping to disk
    """

    pickle_backend = "pickle"  # another optional value is "dill" which can pickle more things of python.
    default_dump_all = False  # if dump all things
    config_attr = ["_include", "_exclude"]
    exclude_attr = []  # exclude_attr have lower priorities than `self._exclude`
    include_attr = []  # include_attr have lower priorities then `self._include`
    FLAG_KEY = "_qlib_serial_flag"

    def __init__(self):
        self._dump_all = self.default_dump_all
        self._exclude = None  # this attribute have higher priorities than `exclude_attr`

    def _is_kept(self, key):
        if key in self.config_attr:
            return False
        if key in self._get_attr_list("include"):
            return True
        if key in self._get_attr_list("exclude"):
            return False
        return self.dump_all or not key.startswith("_")

    def __getstate__(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if self._is_kept(k)}

    def __setstate__(self, state: dict):
        self.__dict__.update(state)

    @property
    def dump_all(self):
        """
        will the object dump all object
        """
        return getattr(self, "_dump_all", False)

    def _get_attr_list(self, attr_type: str) -> list:
        """
        What attribute will not be in specific list

        Parameters
        ----------
        attr_type : str
            "include" or "exclude"

        Returns
        -------
        list:
        """
        if hasattr(self, f"_{attr_type}"):
            res = getattr(self, f"_{attr_type}", [])
        else:
            res = getattr(self.__class__, f"{attr_type}_attr", [])
        if res is None:
            return []
        return res

    def config(self, recursive=False, **kwargs):
        """
        configure the serializable object

        Parameters
        ----------
        kwargs may include following keys

            dump_all : bool
                will the object dump all object
            exclude : list
                What attribute will not be dumped
            include : list
                What attribute will be dumped

        recursive : bool
            will the configuration be recursive
        """
        keys = {"dump_all", "exclude", "include"}
        for k, v in kwargs.items():
            if k in keys:
                attr_name = f"_{k}"
                setattr(self, attr_name, v)
            else:
                raise KeyError(f"Unknown parameter: {k}")

        if recursive:
            for obj in self.__dict__.values():
                # set flag to prevent endless loop
                self.__dict__[self.FLAG_KEY] = True
                if isinstance(obj, Serializable) and self.FLAG_KEY not in obj.__dict__:
                    obj.config(recursive=True, **kwargs)
                del self.__dict__[self.FLAG_KEY]

    def to_pickle(self, path: Union[Path, str], **kwargs):
        """
        Dump self to a pickle file.

        path (Union[Path, str]): the path to dump

        kwargs may include following keys

            dump_all : bool
                will the object dump all object
            exclude : list
                What attribute will not be dumped
            include : list
                What attribute will be dumped
        """
        self.config(**kwargs)
        with Path(path).open("wb") as f:
            # pickle interface like backend; such as dill
            self.get_backend().dump(self, f, protocol=C.dump_protocol_version)

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
        # NOTE: pickle interface like backend; such as dill
        if cls.pickle_backend == "pickle":
            return pickle
        elif cls.pickle_backend == "dill":
            return dill
        else:
            raise ValueError("Unknown pickle backend, please use 'pickle' or 'dill'.")

    @staticmethod
    def general_dump(obj, path: Union[Path, str]):
        """
        A general dumping method for object

        Parameters
        ----------
        obj : object
            the object to be dumped
        path : Union[Path, str]
            the target path the data will be dumped
        """
        path = Path(path)
        if isinstance(obj, Serializable):
            obj.to_pickle(path)
        else:
            with path.open("wb") as f:
                pickle.dump(obj, f, protocol=C.dump_protocol_version)
