# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for backward compatibility after migrating config to Pydantic.

These tests ensure that all existing access patterns continue to work
after the Config/QlibConfig refactor from a raw dict wrapper to a
Pydantic BaseModel-backed implementation.
"""

import pickle

import pytest

from qlib.config import C, QlibConfig, QlibConfigModel


class TestQlibConfigModel:
    """Tests for the Pydantic model itself."""

    def test_model_instantiation(self):
        model = QlibConfigModel()
        assert model.redis_host == "127.0.0.1"
        assert model.redis_port == 6379
        assert model.calendar_provider == "LocalCalendarProvider"

    def test_model_extra_fields(self):
        model = QlibConfigModel(custom_key="custom_value")
        assert model.custom_key == "custom_value"

    def test_model_mutable_defaults_are_independent(self):
        m1 = QlibConfigModel()
        m2 = QlibConfigModel()
        m1.logging_config["test_key"] = "test_value"
        assert "test_key" not in m2.logging_config

    def test_model_dump(self):
        model = QlibConfigModel()
        d = model.model_dump()
        assert isinstance(d, dict)
        assert "redis_host" in d
        assert "logging_config" in d

    def test_model_deep_copy(self):
        model = QlibConfigModel()
        copy = model.model_copy(deep=True)
        copy.redis_host = "changed"
        assert model.redis_host == "127.0.0.1"


class TestConfigDictAccess:
    """Tests for dictionary-style access C['key']."""

    def setup_method(self):
        C.reset()

    def test_getitem_known_key(self):
        assert C["redis_host"] == "127.0.0.1"
        assert C["redis_port"] == 6379
        assert C["provider_uri"] == ""

    def test_getitem_missing_key_raises(self):
        with pytest.raises(KeyError):
            C["nonexistent_key_12345"]

    def test_setitem(self):
        C["redis_host"] = "10.0.0.1"
        assert C["redis_host"] == "10.0.0.1"

    def test_setitem_extra_key(self):
        C["flask_server"] = "0.0.0.0"
        assert C["flask_server"] == "0.0.0.0"


class TestConfigAttrAccess:
    """Tests for attribute-style access C.key."""

    def setup_method(self):
        C.reset()

    def test_getattr_known_key(self):
        assert C.redis_host == "127.0.0.1"
        assert C.calendar_provider == "LocalCalendarProvider"
        assert C.dump_protocol_version == 4

    def test_getattr_missing_key_raises(self):
        with pytest.raises(AttributeError):
            C.nonexistent_key_12345

    def test_setattr(self):
        C.redis_host = "10.0.0.1"
        assert C.redis_host == "10.0.0.1"


class TestConfigGet:
    """Tests for C.get(key, default)."""

    def setup_method(self):
        C.reset()

    def test_get_existing_key(self):
        assert C.get("redis_host") == "127.0.0.1"

    def test_get_missing_key_returns_default(self):
        assert C.get("nonexistent", 42) == 42

    def test_get_missing_key_returns_none(self):
        assert C.get("nonexistent") is None


class TestConfigContains:
    """Tests for 'key in C'."""

    def setup_method(self):
        C.reset()

    def test_contains_known_key(self):
        assert "redis_host" in C
        assert "logging_config" in C

    def test_not_contains_unknown_key(self):
        assert "nonexistent_key_12345" not in C

    def test_contains_after_setting_extra(self):
        C["new_dynamic_key"] = "value"
        assert "new_dynamic_key" in C


class TestConfigUpdate:
    """Tests for C.update()."""

    def setup_method(self):
        C.reset()

    def test_update_dict(self):
        C.update({"redis_host": "192.168.1.1", "redis_port": 6380})
        assert C.redis_host == "192.168.1.1"
        assert C.redis_port == 6380

    def test_update_kwargs(self):
        C.update(redis_host="10.0.0.1")
        assert C.redis_host == "10.0.0.1"


class TestConfigReset:
    """Tests for C.reset()."""

    def test_reset_restores_defaults(self):
        C.redis_host = "changed"
        C["redis_port"] = 9999
        C.reset()
        assert C.redis_host == "127.0.0.1"
        assert C.redis_port == 6379

    def test_reset_removes_extra_keys(self):
        C["temporary_key"] = "value"
        assert "temporary_key" in C
        C.reset()
        assert "temporary_key" not in C


class TestNestedDictMutation:
    """Tests for mutating nested dict fields in-place."""

    def setup_method(self):
        C.reset()

    def test_logging_config_mutation(self):
        C.logging_config["loggers"]["qlib"]["propagate"] = True
        assert C.logging_config["loggers"]["qlib"]["propagate"] is True

    def test_pit_record_type_access(self):
        assert C.pit_record_type["date"] == "I"
        assert C.pit_record_type["value"] == "d"

    def test_pit_record_nan_access(self):
        assert C.pit_record_nan["date"] == 0
        assert C.pit_record_nan["index"] == 0xFFFFFFFF

    def test_mongo_config_access(self):
        assert C.mongo["task_url"] == "mongodb://localhost:27017/"

    def test_exp_manager_nested_access(self):
        assert C.exp_manager["class"] == "MLflowExpManager"
        assert "uri" in C.exp_manager["kwargs"]


class TestClassConstants:
    """Tests for class-level constants."""

    def test_local_uri(self):
        assert C.LOCAL_URI == "local"

    def test_nfs_uri(self):
        assert C.NFS_URI == "nfs"

    def test_default_freq(self):
        assert C.DEFAULT_FREQ == "__DEFAULT_FREQ"


class TestRegisteredProperty:
    """Tests for the registered property."""

    def test_initial_registered_false(self):
        config = QlibConfig()
        assert config.registered is False


class TestPickle:
    """Tests for pickle serialization."""

    def setup_method(self):
        C.reset()

    def test_pickle_roundtrip(self):
        C["redis_host"] = "pickled_host"
        data = pickle.dumps(C)
        C2 = pickle.loads(data)
        assert C2["redis_host"] == "pickled_host"
        assert C2.redis_port == 6379

    def test_pickle_preserves_registered(self):
        data = pickle.dumps(C)
        C2 = pickle.loads(data)
        assert C2.registered is False


class TestStrRepr:
    """Tests for string representation."""

    def setup_method(self):
        C.reset()

    def test_str_is_dict_like(self):
        s = str(C)
        assert "calendar_provider" in s
        assert "LocalCalendarProvider" in s

    def test_repr_is_dict_like(self):
        r = repr(C)
        assert "redis_host" in r


class TestDefaultValues:
    """Tests that all default values match the original _default_config."""

    def setup_method(self):
        C.reset()

    def test_provider_defaults(self):
        assert C.calendar_provider == "LocalCalendarProvider"
        assert C.instrument_provider == "LocalInstrumentProvider"
        assert C.feature_provider == "LocalFeatureProvider"
        assert C.pit_provider == "LocalPITProvider"
        assert C.expression_provider == "LocalExpressionProvider"
        assert C.dataset_provider == "LocalDatasetProvider"
        assert C.provider == "LocalProvider"
        assert C.provider_uri == ""

    def test_cache_defaults(self):
        assert C.expression_cache is None
        assert C.calendar_cache is None
        assert C.local_cache_path is None
        assert C.default_disk_cache == 1
        assert C.mem_cache_size_limit == 500
        assert C.mem_cache_limit_type == "length"
        assert C.mem_cache_expire == 3600
        assert C.dataset_cache_dir_name == "dataset_cache"
        assert C.features_cache_dir_name == "features_cache"

    def test_redis_defaults(self):
        assert C.redis_host == "127.0.0.1"
        assert C.redis_port == 6379
        assert C.redis_task_db == 1
        assert C.redis_password is None

    def test_misc_defaults(self):
        assert C.logging_level == 20  # logging.INFO
        assert C.dump_protocol_version == 4
        assert C.maxtasksperchild is None
        assert C.joblib_backend == "multiprocessing"
        assert C.min_data_shift == 0
