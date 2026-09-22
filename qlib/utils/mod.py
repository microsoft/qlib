# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
All module related class, e.g. :
- importing a module, class
- walkiing a module
- operations on class or module...
"""

import contextlib
import hashlib
import importlib
import os
from pathlib import Path
import pkgutil
import re
import sys
from types import ModuleType
from typing import Any, Dict, List, Tuple, Union
from urllib.parse import urlparse

from qlib.typehint import InstConf
from qlib.utils.pickle_utils import restricted_pickle_load

CONFIG_MIGRATION_GUIDE = (
    "https://qlib.readthedocs.io/en/latest/start/config_migration.html "
    "(source: https://github.com/microsoft/qlib/blob/security/constrain-config-execution/"
    "docs/start/config_migration.rst)"
)


def _validate_module_trust(trusted):
    # isinstance() can accept objects that spoof __class__; consent must be an actual bool.
    if type(trusted) is not bool:  # pylint: disable=unidiomatic-typecheck
        raise TypeError(f"trusted must be a boolean. Migration guide: {CONFIG_MIGRATION_GUIDE}")


def _register_legacy_module_alias(module, module_path, module_file):
    """Keep trusted old model pickles loadable after their module is imported."""
    for path in (module_path, str(module_file)):
        legacy_name = re.sub("^[^a-zA-Z_]+", "", re.sub("[^0-9a-zA-Z_]", "", path[:-3].replace("/", "_")))
        previous = sys.modules.get(legacy_name)
        previous_file = getattr(previous, "__file__", None)
        if legacy_name and (
            previous is None or (previous_file is not None and Path(previous_file).resolve() == module_file)
        ):
            sys.modules[legacy_name] = module


def get_module_by_module_path(module_path: Union[str, ModuleType], *, trusted: bool = False):
    """Load module path

    :param module_path:
    :param trusted: Explicit consent to execute this ``.py`` file. Must be a boolean;
        defaults to ``False``. Package imports remain available. This does not
        restrict directories, sandbox code, or authorize artifact deserialization.
    :return:
    :raises: ModuleNotFoundError
    """
    _validate_module_trust(trusted)
    if module_path is None:
        raise ModuleNotFoundError("None is passed in as parameters as module_path")

    if isinstance(module_path, ModuleType):
        module = module_path
    else:
        if module_path.endswith(".py"):
            if not trusted:
                raise PermissionError(
                    f"Loading Python file {module_path!r} is disabled by default. "
                    "Only after reviewing its code and who can modify it, set trusted: true on this component "
                    "alongside class/module_path (not in kwargs), or pass trusted=True to "
                    "get_module_by_module_path for a direct import. "
                    f"Migration guide: {CONFIG_MIGRATION_GUIDE}"
                )
            module_file = Path(module_path).expanduser().resolve(strict=True)
            if not module_file.is_file() or module_file.suffix.lower() != ".py":
                raise ValueError(f"Module path {str(module_file)!r} must be a Python source file")

            readable_name = re.sub("^[^a-zA-Z_]+", "", re.sub("[^0-9a-zA-Z_]", "", module_file.stem))
            path_digest = hashlib.sha256(str(module_file).encode()).hexdigest()[:12]
            module_name = f"_qlib_file_module_{readable_name}_{path_digest}"
            module_spec = importlib.util.spec_from_file_location(module_name, module_file)
            if module_spec is None or module_spec.loader is None:
                raise ImportError(f"Unable to create a module spec for {str(module_file)!r}")
            module = importlib.util.module_from_spec(module_spec)
            sys.modules[module_name] = module
            try:
                module_spec.loader.exec_module(module)
            except Exception:
                sys.modules.pop(module_name, None)
                raise
            _register_legacy_module_alias(module, module_path, module_file)
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
        A dictionary's top-level ``trusted`` boolean authorizes its file-module import.
        It defaults to ``False`` and is separate from constructor ``kwargs``.
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
        trusted = config.get("trusted", False)
        _validate_module_trust(trusted)
        key = "class" if "class" in config else "func"
        if isinstance(config[key], str):
            # 1) get module and class
            # - case 1): "a.b.c.ClassName"
            # - case 2): {"class": "ClassName", "module_path": "a.b.c"}
            m_path, cls = split_module_path(config[key])
            if m_path == "":
                m_path = config.get("module_path", default_module)
            module = get_module_by_module_path(m_path, trusted=trusted)

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
        File-based components require a top-level ``trusted: True`` in their
        configuration dictionary. Each nested component needs its own consent.
        This flag is not forwarded to the constructor and does not authorize
        pickle loading. Constructor arguments named ``trusted`` still belong in
        ``config["kwargs"]``, ``try_kwargs``, or this function's ``**kwargs``.

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
                # To enable relative path like file://data/a/b/c.pkl.  pr.netloc will be data
                path = pr.path
                if pr.netloc != "":
                    path = path.lstrip("/")

                pr_path = os.path.join(pr.netloc, path) if bool(pr.path) else pr.netloc
                with open(os.path.normpath(pr_path), "rb") as f:
                    return restricted_pickle_load(f)
        else:
            with config.open("rb") as f:
                return restricted_pickle_load(f)

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
