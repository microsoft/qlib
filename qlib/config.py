# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
About the configs
=================

The config will be based on QlibConfigModel (a Pydantic BaseModel).
Two modes are supported
- client
- server

"""
from __future__ import annotations

import os
import re
import copy
import logging
import platform
import multiprocessing
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
from typing import TYPE_CHECKING

from qlib.constant import REG_CN, REG_US, REG_TW

if TYPE_CHECKING:
    from qlib.utils.time import Freq

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MLflowSettings(BaseSettings):
    uri: str = "file:" + str(Path(os.getcwd()).resolve() / "mlruns")
    default_exp_name: str = "Experiment"


class QSettings(BaseSettings):
    """
    Qlib's settings.
    It tries to provide a default settings for most of Qlib's components.
    But it would be a long journey to provide a comprehensive settings for all of Qlib's components.

    Here is some design guidelines:
    - The priority of settings is
        - Actively passed-in settings, like `qlib.init(provider_uri=...)`
        - The default settings
            - QSettings tries to provide default settings for most of Qlib's components.
    """

    mlflow: MLflowSettings = MLflowSettings()
    provider_uri: str = "~/.qlib/qlib_data/cn_data"

    model_config = SettingsConfigDict(
        env_prefix="QLIB_",
        env_nested_delimiter="_",
    )


QSETTINGS = QSettings()


# pickle.dump protocol version: https://docs.python.org/3/library/pickle.html#data-stream-format
PROTOCOL_VERSION = 4

NUM_USABLE_CPU = max(multiprocessing.cpu_count() - 2, 1)

DISK_DATASET_CACHE = "DiskDatasetCache"
SIMPLE_DATASET_CACHE = "SimpleDatasetCache"
DISK_EXPRESSION_CACHE = "DiskExpressionCache"

DEPENDENCY_REDIS_CACHE = (DISK_DATASET_CACHE, DISK_EXPRESSION_CACHE)


class QlibConfigModel(BaseModel):
    """Pydantic model defining all typed configuration fields for Qlib.

    This model provides type annotations, default values, and validation
    for Qlib's configuration. It uses ``extra="allow"`` so that dynamic
    keys (e.g. ``flask_server``, ``region``, ``mount_path``) can be set
    at runtime via ``set_mode()`` or ``set()``.
    """

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    # --- Data provider config ---
    calendar_provider: str = "LocalCalendarProvider"
    instrument_provider: str = "LocalInstrumentProvider"
    feature_provider: str = "LocalFeatureProvider"
    pit_provider: str = "LocalPITProvider"
    expression_provider: str = "LocalExpressionProvider"
    dataset_provider: str = "LocalDatasetProvider"
    provider: str = "LocalProvider"
    # provider_uri can be a str or a dict mapping freq -> uri
    provider_uri: Union[str, Dict[str, str]] = ""

    # --- Cache config ---
    expression_cache: Optional[str] = None
    calendar_cache: Optional[str] = None
    local_cache_path: Optional[Union[str, Path]] = None
    default_disk_cache: int = 1  # 0: skip / 1: use
    mem_cache_size_limit: int = 500
    mem_cache_limit_type: str = "length"
    mem_cache_expire: int = 60 * 60  # 1 hour
    dataset_cache_dir_name: str = "dataset_cache"
    features_cache_dir_name: str = "features_cache"

    # --- Parallel processing config ---
    # kernels can be a fixed int or a callable (freq: str) -> int
    kernels: Union[int, Callable] = NUM_USABLE_CPU
    dump_protocol_version: int = PROTOCOL_VERSION
    maxtasksperchild: Optional[int] = None
    joblib_backend: Optional[str] = "multiprocessing"

    # --- Redis config ---
    redis_host: str = "127.0.0.1"
    redis_port: int = 6379
    redis_task_db: int = 1
    redis_password: Optional[str] = None

    # --- Logging config ---
    logging_level: int = logging.INFO
    logging_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            "version": 1,
            "formatters": {
                "logger_format": {
                    "format": "[%(process)s:%(threadName)s](%(asctime)s) %(levelname)s"
                    " - %(name)s - [%(filename)s:%(lineno)d] - %(message)s"
                }
            },
            "filters": {
                "field_not_found": {
                    "()": "qlib.log.LogFilter",
                    "param": [".*?WARN: data not found for.*?"],
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": logging.DEBUG,
                    "formatter": "logger_format",
                    "filters": ["field_not_found"],
                }
            },
            "loggers": {"qlib": {"level": logging.DEBUG, "handlers": ["console"], "propagate": False}},
            "disable_existing_loggers": False,
        }
    )

    # --- Experiment manager config ---
    exp_manager: Dict[str, Any] = Field(
        default_factory=lambda: {
            "class": "MLflowExpManager",
            "module_path": "qlib.workflow.expm",
            "kwargs": {
                "uri": QSETTINGS.mlflow.uri,
                "default_exp_name": QSETTINGS.mlflow.default_exp_name,
            },
        }
    )

    # --- PIT record config ---
    pit_record_type: Dict[str, str] = Field(
        default_factory=lambda: {
            "date": "I",  # uint32
            "period": "I",  # uint32
            "value": "d",  # float64
            "index": "I",  # uint32
        }
    )
    pit_record_nan: Dict[str, Any] = Field(
        default_factory=lambda: {
            "date": 0,
            "period": 0,
            "value": float("NAN"),
            "index": 0xFFFFFFFF,
        }
    )

    # --- MongoDB config ---
    mongo: Dict[str, str] = Field(
        default_factory=lambda: {
            "task_url": "mongodb://localhost:27017/",
            "task_db_name": "default_task_db",
        }
    )

    # --- Misc ---
    min_data_shift: int = 0


class Config:
    """Backward-compatible config wrapper around :class:`QlibConfigModel`.

    Supports both dictionary-style (``C["key"]``) and attribute-style
    (``C.key``) access, as well as ``update()``, ``reset()``, and pickle
    serialization.
    """

    def __init__(self, default_conf=None):
        # Store in __dict__ directly to avoid triggering __setattr__
        self.__dict__["_default_model"] = QlibConfigModel()
        if default_conf is not None:
            # Apply the legacy default_conf dict as initial overrides
            self.__dict__["_default_model"] = QlibConfigModel(**default_conf)
        self.__dict__["_model"] = self.__dict__["_default_model"].model_copy(deep=True)

    def __getitem__(self, key):
        try:
            return getattr(self.__dict__["_model"], key)
        except AttributeError:
            raise KeyError(key)

    def __getattr__(self, attr):
        try:
            return getattr(self.__dict__["_model"], attr)
        except AttributeError:
            raise AttributeError(f"No such `{attr}` in self._config")

    def get(self, key, default=None):
        return getattr(self.__dict__["_model"], key, default)

    def __setitem__(self, key, value):
        setattr(self.__dict__["_model"], key, value)

    def __setattr__(self, attr, value):
        setattr(self.__dict__["_model"], attr, value)

    def __contains__(self, item):
        return hasattr(self.__dict__["_model"], item)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __str__(self):
        return str(self.__dict__["_model"].model_dump())

    def __repr__(self):
        return str(self.__dict__["_model"].model_dump())

    def reset(self):
        self.__dict__["_model"] = self.__dict__["_default_model"].model_copy(deep=True)

    def update(self, *args, **kwargs):
        d = dict(*args, **kwargs)
        for k, v in d.items():
            setattr(self.__dict__["_model"], k, v)

    def set_conf_from_C(self, config_c):
        model = config_c.__dict__["_model"]
        for k, v in model.model_dump().items():
            setattr(self.__dict__["_model"], k, v)
        # Also copy extra fields
        if hasattr(model, "__pydantic_extra__") and model.__pydantic_extra__:
            for k, v in model.__pydantic_extra__.items():
                setattr(self.__dict__["_model"], k, v)

    @staticmethod
    def register_from_C(config, skip_register=True):
        from .utils import set_log_with_config  # pylint: disable=C0415

        if C.registered and skip_register:
            return

        C.set_conf_from_C(config)
        if C.logging_config:
            set_log_with_config(C.logging_config)
        C.register()


MODE_CONF = {
    "server": {
        # config it in qlib.init()
        "provider_uri": "",
        # redis
        "redis_host": "127.0.0.1",
        "redis_port": 6379,
        "redis_task_db": 1,
        # cache
        "expression_cache": DISK_EXPRESSION_CACHE,
        "dataset_cache": DISK_DATASET_CACHE,
        "local_cache_path": Path("~/.cache/qlib_simple_cache").expanduser().resolve(),
        "mount_path": None,
    },
    "client": {
        # config it in user's own code
        "provider_uri": QSETTINGS.provider_uri,
        # cache
        # Using parameter 'remote' to announce the client is using server_cache, and the writing access will be disabled.
        # Disable cache by default. Avoid introduce advanced features for beginners
        "dataset_cache": None,
        # SimpleDatasetCache directory
        "local_cache_path": Path("~/.cache/qlib_simple_cache").expanduser().resolve(),
        # client config
        "mount_path": None,
        "auto_mount": False,  # The nfs is already mounted on our server[auto_mount: False].
        # The nfs should be auto-mounted by qlib on other
        # serversS(such as PAI) [auto_mount:True]
        "timeout": 100,
        "logging_level": logging.INFO,
        "region": REG_CN,
        # custom operator
        # each element of custom_ops should be Type[ExpressionOps] or dict
        # if element of custom_ops is Type[ExpressionOps], it represents the custom operator class
        # if element of custom_ops is dict, it represents the config of custom operator and should include `class` and `module_path` keys.
        "custom_ops": [],
    },
}

HIGH_FREQ_CONFIG = {
    "provider_uri": "~/.qlib/qlib_data/cn_data_1min",
    "dataset_cache": None,
    "expression_cache": "DiskExpressionCache",
    "region": REG_CN,
}

_default_region_config = {
    REG_CN: {
        "trade_unit": 100,
        "limit_threshold": 0.095,
        "deal_price": "close",
    },
    REG_US: {
        "trade_unit": 1,
        "limit_threshold": None,
        "deal_price": "close",
    },
    REG_TW: {
        "trade_unit": 1000,
        "limit_threshold": 0.1,
        "deal_price": "close",
    },
}


class QlibConfig(Config):
    # URI_TYPE
    LOCAL_URI = "local"
    NFS_URI = "nfs"
    DEFAULT_FREQ = "__DEFAULT_FREQ"

    def __init__(self, default_conf=None):
        super().__init__(default_conf)
        self.__dict__["_registered"] = False

    class DataPathManager:
        """
        Motivation:
        - get the right path (e.g. data uri) for accessing data based on given information(e.g. provider_uri, mount_path and frequency)
        - some helper functions to process uri.
        """

        def __init__(self, provider_uri: Union[str, Path, dict], mount_path: Union[str, Path, dict]):
            """
            The relation of `provider_uri` and `mount_path`
            - `mount_path` is used only if provider_uri is an NFS path
            - otherwise, provider_uri will be used for accessing data
            """
            self.provider_uri = provider_uri
            self.mount_path = mount_path

        @staticmethod
        def format_provider_uri(provider_uri: Union[str, dict, Path]) -> dict:
            if provider_uri is None:
                raise ValueError("provider_uri cannot be None")
            if isinstance(provider_uri, (str, dict, Path)):
                if not isinstance(provider_uri, dict):
                    provider_uri = {QlibConfig.DEFAULT_FREQ: provider_uri}
            else:
                raise TypeError(f"provider_uri does not support {type(provider_uri)}")
            for freq, _uri in provider_uri.items():
                if QlibConfig.DataPathManager.get_uri_type(_uri) == QlibConfig.LOCAL_URI:
                    provider_uri[freq] = str(Path(_uri).expanduser().resolve())
            return provider_uri

        @staticmethod
        def get_uri_type(uri: Union[str, Path]):
            uri = uri if isinstance(uri, str) else str(uri.expanduser().resolve())
            is_win = re.match("^[a-zA-Z]:.*", uri) is not None  # such as 'C:\\data', 'D:'
            # such as 'host:/data/'   (User may define short hostname by themselves or use localhost)
            is_nfs_or_win = re.match("^[^/]+:.+", uri) is not None

            if is_nfs_or_win and not is_win:
                return QlibConfig.NFS_URI
            else:
                return QlibConfig.LOCAL_URI

        def get_data_uri(self, freq: Optional[Union[str, Freq]] = None) -> Path:
            """
            please refer DataPathManager's __init__ and class doc
            """
            if freq is not None:
                freq = str(freq)  # converting Freq to string
            if freq is None or freq not in self.provider_uri:
                freq = QlibConfig.DEFAULT_FREQ
            _provider_uri = self.provider_uri[freq]
            if self.get_uri_type(_provider_uri) == QlibConfig.LOCAL_URI:
                return Path(_provider_uri)
            elif self.get_uri_type(_provider_uri) == QlibConfig.NFS_URI:
                if "win" in platform.system().lower():
                    # windows, mount_path is the drive
                    _path = str(self.mount_path[freq])
                    return Path(f"{_path}:\\") if ":" not in _path else Path(_path)
                return Path(self.mount_path[freq])
            else:
                raise NotImplementedError(f"This type of uri is not supported")

    def set_mode(self, mode):
        # raise KeyError
        self.update(MODE_CONF[mode])
        # TODO: update region based on kwargs

    def set_region(self, region):
        # raise KeyError
        self.update(_default_region_config[region])

    @staticmethod
    def is_depend_redis(cache_name: str):
        return cache_name in DEPENDENCY_REDIS_CACHE

    @property
    def dpm(self):
        return self.DataPathManager(self["provider_uri"], self["mount_path"])

    def resolve_path(self):
        # resolve path
        _mount_path = self["mount_path"]
        _provider_uri = self.DataPathManager.format_provider_uri(self["provider_uri"])
        if not isinstance(_mount_path, dict):
            _mount_path = {_freq: _mount_path for _freq in _provider_uri.keys()}

        # check provider_uri and mount_path
        _miss_freq = set(_provider_uri.keys()) - set(_mount_path.keys())
        assert len(_miss_freq) == 0, f"mount_path is missing freq: {_miss_freq}"

        # resolve
        for _freq in _provider_uri.keys():
            # mount_path
            _mount_path[_freq] = (
                _mount_path[_freq] if _mount_path[_freq] is None else str(Path(_mount_path[_freq]).expanduser())
            )
        self["provider_uri"] = _provider_uri
        self["mount_path"] = _mount_path

    def set(self, default_conf: str = "client", **kwargs):
        """
        configure qlib based on the input parameters

        The configuration will act like a dictionary.

        Normally, it literally is replaced the value according to the keys.
        However, sometimes it is hard for users to set the config when the configuration is nested and complicated

        So this API provides some special parameters for users to set the keys in a more convenient way.
        - region:  REG_CN, REG_US
            - several region-related config will be changed

        Parameters
        ----------
        default_conf : str
            the default config template chosen by user: "server", "client"
        """
        from .utils import set_log_with_config, get_module_logger, can_use_cache  # pylint: disable=C0415

        self.reset()

        _logging_config = kwargs.get("logging_config", self.logging_config)

        # set global config
        if _logging_config:
            set_log_with_config(_logging_config)

        logger = get_module_logger("Initialization", kwargs.get("logging_level", self.logging_level))
        logger.info(f"default_conf: {default_conf}.")

        self.set_mode(default_conf)
        self.set_region(kwargs.get("region", self["region"] if "region" in self else REG_CN))

        for k, v in kwargs.items():
            if k not in self:
                logger.warning("Unrecognized config %s" % k)
            self[k] = v

        self.resolve_path()

        if not (self["expression_cache"] is None and self["dataset_cache"] is None):
            # check redis
            if not can_use_cache():
                log_str = ""
                # check expression cache
                if self.is_depend_redis(self["expression_cache"]):
                    log_str += self["expression_cache"]
                    self["expression_cache"] = None
                # check dataset cache
                if self.is_depend_redis(self["dataset_cache"]):
                    log_str += f" and {self['dataset_cache']}" if log_str else self["dataset_cache"]
                    self["dataset_cache"] = None
                if log_str:
                    logger.warning(
                        f"redis connection failed(host={self['redis_host']} port={self['redis_port']}), "
                        f"{log_str} will not be used!"
                    )

    def register(self):
        from .utils import init_instance_by_config  # pylint: disable=C0415
        from .data.ops import register_all_ops  # pylint: disable=C0415
        from .data.data import register_all_wrappers  # pylint: disable=C0415
        from .workflow import R, QlibRecorder  # pylint: disable=C0415
        from .workflow.utils import experiment_exit_handler  # pylint: disable=C0415

        register_all_ops(self)
        register_all_wrappers(self)
        # set up QlibRecorder
        exp_manager = init_instance_by_config(self["exp_manager"])
        qr = QlibRecorder(exp_manager)
        R.register(qr)
        # clean up experiment when python program ends
        experiment_exit_handler()

        # Supporting user reset qlib version (useful when user want to connect to qlib server with old version)
        self.reset_qlib_version()

        self.__dict__["_registered"] = True

    def reset_qlib_version(self):
        import qlib  # pylint: disable=C0415

        reset_version = self.get("qlib_reset_version", None)
        if reset_version is not None:
            qlib.__version__ = reset_version
        else:
            qlib.__version__ = getattr(qlib, "__version__bak")
            # Due to a bug? that converting __version__ to _QlibConfig__version__bak
            # Using  __version__bak instead of __version__

    def get_kernels(self, freq: str):
        """get number of processors given frequency"""
        if isinstance(self["kernels"], Callable):
            return self["kernels"](freq)
        return self["kernels"]

    @property
    def registered(self):
        return self.__dict__["_registered"]


# global config
C = QlibConfig()
