# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
All module related class, e.g. :
- importing a module, class
- walkiing a module
- operations on class or module...
"""

import contextlib
import importlib
import os
from pathlib import Path
import pickle
import pkgutil
import re
import sys
from types import ModuleType
from typing import Any, Dict, List, Tuple, Union
from urllib.parse import urlparse

from qlib.typehint import InstConf


def get_module_by_module_path(module_path: Union[str, ModuleType]):
    """Load module path

    :param module_path:
    :return:
    :raises: ModuleNotFoundError
    """
    if module_path is None:
        raise ModuleNotFoundError("None is passed in as parameters as module_path")

    if isinstance(module_path, ModuleType):
        module = module_path
    else:
        if module_path.endswith(".py"):
            module_name = re.sub("^[^a-zA-Z_]+", "", re.sub("[^0-9a-zA-Z_]", "", module_path[:-3].replace("/", "_")))
            module_spec = importlib.util.spec_from_file_location(module_name, module_path)
            module = importlib.util.module_from_spec(module_spec)
            sys.modules[module_name] = module
            module_spec.loader.exec_module(module)
        else:
            module = importlib.import_module(module_path)
    return module


def split_module_path(module_path: str) -> Tuple[str, str]:
    """

    Parameters
    ----------
    module_path : str
        e.g. "a.b.c.ClassName"

    Returns
    -------
    Tuple[str, str]
        e.g. ("a.b.c", "ClassName")
    """
    *m_path, cls = module_path.split(".")
    m_path = ".".join(m_path)
    return m_path, cls


def get_callable_kwargs(config: InstConf, default_module: Union[str, ModuleType] = None) -> (type, dict):
    """
    extract class/func and kwargs from config info

    Parameters
    ----------
    config : [dict, str]
        similar to config
        please refer to the doc of init_instance_by_config

    default_module : Python module or str
        It should be a python module to load the class type
        This function will load class from the config['module_path'] first.
        If config['module_path'] doesn't exists, it will load the class from default_module.

    Returns
    -------
    (type, dict):
        the class/func object and it's arguments.

    Raises
    ------
        ModuleNotFoundError
    """
    if isinstance(config, dict):
        key = "class" if "class" in config else "func"
        if isinstance(config[key], str):
            # 1) get module and class
            # - case 1): "a.b.c.ClassName"
            # - case 2): {"class": "ClassName", "module_path": "a.b.c"}
            m_path, cls = split_module_path(config[key])
            if m_path == "":
                m_path = config.get("module_path", default_module)
            module = get_module_by_module_path(m_path)

            # 2) get callable
            _callable = getattr(module, cls)  # may raise AttributeError
        else:
            _callable = config[key]  # the class type itself is passed in
        kwargs = config.get("kwargs", {})
    elif isinstance(config, str):
        # a.b.c.ClassName
        m_path, cls = split_module_path(config)
        module = get_module_by_module_path(default_module if m_path == "" else m_path)

        _callable = getattr(module, cls)
        kwargs = {}
    else:
        raise NotImplementedError(f"This type of input is not supported")
    return _callable, kwargs


get_cls_kwargs = get_callable_kwargs  # NOTE: this is for compatibility for the previous version


def init_instance_by_config(
    config: InstConf,
    default_module=None,
    accept_types: Union[type, Tuple[type]] = (),
    try_kwargs: Dict = {},
    **kwargs,
) -> Any:
    """
    get initialized instance with config

    Parameters
    ----------
    config : InstConf

    default_module : Python module
        Optional. It should be a python module.
        NOTE: the "module_path" will be override by `module` arguments

        This function will load class from the config['module_path'] first.
        If config['module_path'] doesn't exists, it will load the class from default_module.

    accept_types: Union[type, Tuple[type]]
        Optional. If the config is a instance of specific type, return the config directly.
        This will be passed into the second parameter of isinstance.

    try_kwargs: Dict
        Try to pass in kwargs in `try_kwargs` when initialized the instance
        If error occurred, it will fail back to initialization without try_kwargs.

    Returns
    -------
    object:
        An initialized object based on the config info
    """
    if isinstance(config, accept_types):
        return config

    if isinstance(config, (str, Path)):
        if isinstance(config, str):
            # path like 'file:///<path to pickle file>/obj.pkl'
            pr = urlparse(config)
            if pr.scheme == "file":
                pr_path = os.path.join(pr.netloc, pr.path) if bool(pr.path) else pr.netloc
                with open(os.path.normpath(pr_path), "rb") as f:
                    return pickle.load(f)
        else:
            with config.open("rb") as f:
                return pickle.load(f)

    klass, cls_kwargs = get_callable_kwargs(config, default_module=default_module)

    try:
        return klass(**cls_kwargs, **try_kwargs, **kwargs)
    except (TypeError,):
        # TypeError for handling errors like
        # 1: `XXX() got multiple values for keyword argument 'YYY'`
        # 2: `XXX() got an unexpected keyword argument 'YYY'
        return klass(**cls_kwargs, **kwargs)


@contextlib.contextmanager
def class_casting(obj: object, cls: type):
    """
    Python doesn't provide the downcasting mechanism.
    We use the trick here to downcast the class

    Parameters
    ----------
    obj : object
        the object to be cast
    cls : type
        the target class type
    """
    orig_cls = obj.__class__
    obj.__class__ = cls
    yield
    obj.__class__ = orig_cls


def find_all_classes(module_path: Union[str, ModuleType], cls: type) -> List[type]:
    """
    Find all the classes recursively that inherit from `cls` in a given module.
    - `cls` itself is also included

        >>> from qlib.data.dataset.handler import DataHandler
        >>> find_all_classes("qlib.contrib.data.handler", DataHandler)
        [<class 'qlib.contrib.data.handler.Alpha158'>, <class 'qlib.contrib.data.handler.Alpha158vwap'>, <class 'qlib.contrib.data.handler.Alpha360'>, <class 'qlib.contrib.data.handler.Alpha360vwap'>, <class 'qlib.data.dataset.handler.DataHandlerLP'>]

    TODO:
    - skip import error

    """
    if isinstance(module_path, ModuleType):
        mod = module_path
    else:
        mod = importlib.import_module(module_path)

    cls_list = []

    def _append_cls(obj):
        # Leverage the closure trick to reuse code
        if isinstance(obj, type) and issubclass(obj, cls) and cls not in cls_list:
            cls_list.append(obj)

    for attr in dir(mod):
        _append_cls(getattr(mod, attr))

    if hasattr(mod, "__path__"):
        # if the model is a package
        for _, modname, _ in pkgutil.iter_modules(mod.__path__):
            sub_mod = importlib.import_module(f"{mod.__package__}.{modname}")
            for m_cls in find_all_classes(sub_mod, cls):
                _append_cls(m_cls)
    return cls_list
