# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


# REGION CONST
REG_CN = "cn"
REG_US = "US"

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
    "kernels": 16,
    # How many tasks belong to one process. Recommend 1 for high-frequency data and None for daily data.
    "maxtasksperchild": None,
    "default_disk_cache": 1,  # 0:skip/1:use
    "disable_disk_cache": False,  # disable disk cache; if High-frequency data generally disable_disk_cache=True
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
    "logging_level": "INFO",
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
                "level": "DEBUG",
                "formatter": "logger_format",
                "filters": ["field_not_found"],
            }
        },
        "loggers": {"qlib": {"level": "DEBUG", "handlers": ["console"]}},
    },
}

_default_server_config = {
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
    "kernels": 64,
    # cache
    "expression_cache": "DiskExpressionCache",
    "dataset_cache": "DiskDatasetCache",
}

_default_client_config = {
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
    "kernels": 16,
    "mount_path": "~/.qlib/qlib_data/cn_data",
    "auto_mount": False,  # The nfs is already mounted on our server[auto_mount: False].
    # The nfs should be auto-mounted by qlib on other
    # serversS(such as PAI) [auto_mount:True]
    "timeout": 100,
    "logging_level": "INFO",
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


class Config:
    def __getitem__(self, key):
        return _default_config[key]

    def __getattr__(self, attr):
        try:
            return _default_config[attr]
        except KeyError:
            return AttributeError(f"No such {attr} in _default_config")

    def __setitem__(self, key, value):
        _default_config[key] = value

    def __setattr__(self, attr, value):
        _default_config[attr] = value

    def __contains__(self, item):
        return item in _default_config

    def __getstate__(self):
        return _default_config

    def __setstate__(self, state):
        _default_config.update(state)

    def __str__(self):
        return str(_default_config)

    def __repr__(self):
        return str(_default_config)


# global config
C = Config()
