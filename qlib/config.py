# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
About the configs
=================

The config will based on _default_config.
Two modes are supported
- client
- server

"""

import os
import re
import copy
import logging
import multiprocessing
from pathlib import Path


class Config:
    def __init__(self, default_conf):
        self.__dict__["_default_config"] = copy.deepcopy(default_conf)  # avoiding conflictions with __getattr__
        self.reset()

    def __getitem__(self, key):
        return self.__dict__["_config"][key]

    def __getattr__(self, attr):
        if attr in self.__dict__["_config"]:
            return self.__dict__["_config"][attr]

        raise AttributeError(f"No such {attr} in self._config")

    def get(self, key, default=None):
        return self.__dict__["_config"].get(key, default)

    def __setitem__(self, key, value):
        self.__dict__["_config"][key] = value

    def __setattr__(self, attr, value):
        self.__dict__["_config"][attr] = value

    def __contains__(self, item):
        return item in self.__dict__["_config"]

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __str__(self):
        return str(self.__dict__["_config"])

    def __repr__(self):
        return str(self.__dict__["_config"])

    def reset(self):
        self.__dict__["_config"] = copy.deepcopy(self._default_config)

    def update(self, *args, **kwargs):
        self.__dict__["_config"].update(*args, **kwargs)

    def set_conf_from_C(self, config_c):
        self.update(**config_c.__dict__["_config"])


# REGION CONST
REG_CN = "cn"
REG_US = "us"

NUM_USABLE_CPU = max(multiprocessing.cpu_count() - 2, 1)

_default_config = {
    # data provider config
    "calendar_provider": "LocalCalendarProvider",
    "instrument_provider": "LocalInstrumentProvider",
    "feature_provider": "LocalFeatureProvider",
    "expression_provider": "LocalExpressionProvider",
    "dataset_provider": "LocalDatasetProvider",
    "provider": "LocalProvider",
    # config it in qlib.init()
    "provider_uri": "",
    # cache
    "expression_cache": None,
    "dataset_cache": None,
    "calendar_cache": None,
    # for simple dataset cache
    "local_cache_path": None,
    "kernels": NUM_USABLE_CPU,
    # How many tasks belong to one process. Recommend 1 for high-frequency data and None for daily data.
    "maxtasksperchild": None,
    "default_disk_cache": 1,  # 0:skip/1:use
    "mem_cache_size_limit": 500,
    # memory cache expire second, only in used 'DatasetURICache' and 'client D.calendar'
    # default 1 hour
    "mem_cache_expire": 60 * 60,
    # memory cache space limit, default 5GB, only in used client
    "mem_cache_space_limit": 1024 * 1024 * 1024 * 5,
    # cache dir name
    "dataset_cache_dir_name": "dataset_cache",
    "features_cache_dir_name": "features_cache",
    # redis
    # in order to use cache
    "redis_host": "127.0.0.1",
    "redis_port": 6379,
    "redis_task_db": 1,
    # This value can be reset via qlib.init
    "logging_level": logging.INFO,
    # Global configuration of qlib log
    # logging_level can control the logging level more finely
    "logging_config": {
        "version": 1,
        "formatters": {
            "logger_format": {
                "format": "[%(process)s:%(threadName)s](%(asctime)s) %(levelname)s - %(name)s - [%(filename)s:%(lineno)d] - %(message)s"
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
        "loggers": {"qlib": {"level": logging.DEBUG, "handlers": ["console"]}},
    },
    # Default config for experiment manager
    "exp_manager": {
        "class": "MLflowExpManager",
        "module_path": "qlib.workflow.expm",
        "kwargs": {
            "uri": "file:" + str(Path(os.getcwd()).resolve() / "mlruns"),
            "default_exp_name": "Experiment",
        },
    },
    # Default config for MongoDB
    "mongo": {
        "task_url": "mongodb://localhost:27017/",
        "task_db_name": "default_task_db",
    },
}

MODE_CONF = {
    "server": {
        # data provider config
        "calendar_provider": "LocalCalendarProvider",
        "instrument_provider": "LocalInstrumentProvider",
        "feature_provider": "LocalFeatureProvider",
        "expression_provider": "LocalExpressionProvider",
        "dataset_provider": "LocalDatasetProvider",
        "provider": "LocalProvider",
        # config it in qlib.init()
        "provider_uri": "",
        # redis
        "redis_host": "127.0.0.1",
        "redis_port": 6379,
        "redis_task_db": 1,
        "kernels": NUM_USABLE_CPU,
        # cache
        "expression_cache": "DiskExpressionCache",
        "dataset_cache": "DiskDatasetCache",
        "mount_path": None,
    },
    "client": {
        # data provider config
        "calendar_provider": "LocalCalendarProvider",
        "instrument_provider": "LocalInstrumentProvider",
        "feature_provider": "LocalFeatureProvider",
        "expression_provider": "LocalExpressionProvider",
        "dataset_provider": "LocalDatasetProvider",
        "provider": "LocalProvider",
        # config it in user's own code
        "provider_uri": "~/.qlib/qlib_data/cn_data",
        # cache
        # Using parameter 'remote' to announce the client is using server_cache, and the writing access will be disabled.
        "expression_cache": "DiskExpressionCache",
        "dataset_cache": "DiskDatasetCache",
        "calendar_cache": None,
        # client config
        "kernels": NUM_USABLE_CPU,
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
    "provider_uri": "~/.qlib/qlib_data/yahoo_cn_1min",
    "dataset_cache": None,
    "expression_cache": "DiskExpressionCache",
    "region": REG_CN,
}

_default_region_config = {
    REG_CN: {
        "trade_unit": 100,
        "limit_threshold": 0.099,
        "deal_price": "vwap",
    },
    REG_US: {
        "trade_unit": 1,
        "limit_threshold": None,
        "deal_price": "close",
    },
}


class QlibConfig(Config):
    # URI_TYPE
    LOCAL_URI = "local"
    NFS_URI = "nfs"

    def __init__(self, default_conf):
        super().__init__(default_conf)
        self._registered = False

    def set_mode(self, mode):
        # raise KeyError
        self.update(MODE_CONF[mode])
        # TODO: update region based on kwargs

    def set_region(self, region):
        # raise KeyError
        self.update(_default_region_config[region])

    def resolve_path(self):
        # resolve path
        if self["mount_path"] is not None:
            self["mount_path"] = str(Path(self["mount_path"]).expanduser().resolve())

        if self.get_uri_type() == QlibConfig.LOCAL_URI:
            self["provider_uri"] = str(Path(self["provider_uri"]).expanduser().resolve())

    def get_uri_type(self):
        is_win = re.match("^[a-zA-Z]:.*", self["provider_uri"]) is not None  # such as 'C:\\data', 'D:'
        is_nfs_or_win = (
            re.match("^[^/]+:.+", self["provider_uri"]) is not None
        )  # such as 'host:/data/'   (User may define short hostname by themselves or use localhost)

        if is_nfs_or_win and not is_win:
            return QlibConfig.NFS_URI
        else:
            return QlibConfig.LOCAL_URI

    def get_data_path(self):
        if self.get_uri_type() == QlibConfig.LOCAL_URI:
            return self["provider_uri"]
        elif self.get_uri_type() == QlibConfig.NFS_URI:
            return self["mount_path"]
        else:
            raise NotImplementedError(f"This type of uri is not supported")

    def set(self, default_conf="client", **kwargs):
        from .utils import set_log_with_config, get_module_logger, can_use_cache

        self.reset()

        _logging_config = self.logging_config
        if "logging_config" in kwargs:
            _logging_config = kwargs["logging_config"]

        # set global config
        if _logging_config:
            set_log_with_config(_logging_config)

        # FIXME: this logger ignored the level in config
        logger = get_module_logger("Initialization", level=logging.INFO)
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
                logger.warning(
                    f"redis connection failed(host={self['redis_host']} port={self['redis_port']}), cache will not be used!"
                )
                self["expression_cache"] = None
                self["dataset_cache"] = None

    def register(self):
        from .utils import init_instance_by_config
        from .data.ops import register_all_ops
        from .data.data import register_all_wrappers
        from .workflow import R, QlibRecorder
        from .workflow.utils import experiment_exit_handler

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

        self._registered = True

    def reset_qlib_version(self):
        import qlib

        reset_version = self.get("qlib_reset_version", None)
        if reset_version is not None:
            qlib.__version__ = reset_version
        else:
            qlib.__version__ = getattr(qlib, "__version__bak")
            # Due to a bug? that converting __version__ to _QlibConfig__version__bak
            # Using  __version__bak instead of __version__

    @property
    def registered(self):
        return self._registered


# global config
C = QlibConfig(_default_config)
