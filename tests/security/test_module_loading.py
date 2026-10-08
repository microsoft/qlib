# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import copy
import json
import multiprocessing
import os
import pickle
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from qlib.utils.mod import (
    CONFIG_MIGRATION_GUIDE,
    get_callable_kwargs,
    get_module_by_module_path,
    init_instance_by_config,
)


@pytest.fixture
def file_component(tmp_path):
    path = tmp_path / "model.py"
    marker = tmp_path / "executed.txt"
    path.write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('executed')\n"
        "class Model:\n"
        "    def __init__(self, value=42, trusted=False, allowed_module_roots=None):\n"
        "        self.value = value\n"
        "        self.trusted = trusted\n"
        "        self.allowed_module_roots = allowed_module_roots\n"
        "def factory(**kwargs):\n"
        "    return Model(**kwargs)\n",
        encoding="utf-8",
    )
    return {"class": "Model", "module_path": str(path)}, marker


@pytest.mark.parametrize("options", [{}, {"trusted": False}])
def test_file_module_loading_refuses_before_execution(file_component, options):
    config, marker = file_component
    with pytest.raises(PermissionError, match="disabled by default") as error:
        get_module_by_module_path(config["module_path"], **options)
    assert not marker.exists()
    assert repr(config["module_path"]) in str(error.value)
    assert "alongside class/module_path (not in kwargs)" in str(error.value)
    assert "get_module_by_module_path" in str(error.value)
    assert CONFIG_MIGRATION_GUIDE in str(error.value)


@pytest.mark.parametrize(
    "module_path",
    [r"C:\custom modules\model.py", r"\\server\share\model.py", "custom/model's.py", "custom/model\nname.py"],
)
@pytest.mark.parametrize("options", [{}, {"trusted": False}])
def test_file_module_refusal_escapes_path(module_path, options):
    with pytest.raises(PermissionError, match="disabled by default") as error:
        get_module_by_module_path(module_path, **options)
    assert str(error.value).startswith(f"Loading Python file {module_path!r} is disabled by default.")


def test_file_module_loading_accepts_explicit_trust(file_component):
    config, marker = file_component
    module = get_module_by_module_path(config["module_path"], trusted=True)
    assert module.Model().value == 42
    assert marker.exists()
    marker.unlink()
    with pytest.raises(PermissionError):
        get_module_by_module_path(config["module_path"])
    assert not marker.exists()


@pytest.mark.parametrize("trusted", [None, "true", "false", 0, 1, [], {}])
@pytest.mark.parametrize("entrypoint", ["direct", "config", "wrapper"])
def test_trust_requires_actual_booleans_before_execution(file_component, trusted, entrypoint):
    from qlib.utils import Wrapper, register_wrapper

    config, marker = file_component
    with pytest.raises(TypeError, match="trusted must be a boolean"):
        if entrypoint == "direct":
            get_module_by_module_path(config["module_path"], trusted=trusted)
        elif entrypoint == "config":
            init_instance_by_config(dict(config, trusted=trusted))
        else:
            register_wrapper(Wrapper(), "Model", config["module_path"], trusted=trusted)
    assert not marker.exists()


def test_trust_validation_does_not_invoke_truthiness(file_component):
    class NotABoolean:
        def __bool__(self):
            raise AssertionError("must not coerce arbitrary values")

    config, marker = file_component
    with pytest.raises(TypeError, match="boolean"):
        init_instance_by_config(dict(config, trusted=NotABoolean()))
    assert not marker.exists()


def test_trust_validation_rejects_spoofed_boolean_class(file_component):
    class NotABoolean:
        @property
        def __class__(self):
            return bool

        def __bool__(self):
            raise AssertionError("must not coerce arbitrary values")

    config, marker = file_component
    trusted = NotABoolean()
    assert isinstance(trusted, bool)
    with pytest.raises(TypeError, match="boolean"):
        get_module_by_module_path(config["module_path"], trusted=trusted)
    with pytest.raises(TypeError, match="boolean"):
        init_instance_by_config(dict(config, trusted=trusted))
    assert not marker.exists()


@pytest.mark.parametrize("key", ["class", "func"])
def test_component_trust_is_metadata_not_a_constructor_argument(file_component, key):
    config, marker = file_component
    config = {key: "Model" if key == "class" else "factory", "module_path": config["module_path"], "trusted": True}
    config["kwargs"] = {"value": 7}
    original = copy.deepcopy(config)
    constructor, kwargs = get_callable_kwargs(config)
    assert kwargs == {"value": 7}
    assert constructor(**kwargs).trusted is False
    assert init_instance_by_config(config).value == 7
    assert config == original
    assert marker.exists()


@pytest.mark.parametrize("location", ["kwargs", "direct", "try_kwargs"])
@pytest.mark.parametrize("trusted", [False, True])
def test_factory_preserves_constructor_trust(file_component, location, trusted):
    config, _ = file_component
    config["trusted"] = True
    options = {}
    if location == "kwargs":
        config["kwargs"] = {"trusted": trusted}
    elif location == "direct":
        options["trusted"] = trusted
    else:
        options["try_kwargs"] = {"trusted": trusted}
    assert init_instance_by_config(config, **options).trusted is trusted


@pytest.mark.parametrize("location", ["kwargs", "direct", "try_kwargs"])
def test_constructor_trust_cannot_authorize_module_import(file_component, location):
    config, marker = file_component
    options = {}
    if location == "kwargs":
        config["kwargs"] = {"trusted": True}
    elif location == "direct":
        options["trusted"] = True
    else:
        options["try_kwargs"] = {"trusted": True}
    with pytest.raises(PermissionError):
        init_instance_by_config(config, **options)
    assert not marker.exists()


def test_factory_does_not_reserve_old_root_constructor_argument(file_component):
    config, _ = file_component
    config["trusted"] = True
    assert init_instance_by_config(config, allowed_module_roots=["constructor value"]).allowed_module_roots == [
        "constructor value"
    ]


@pytest.mark.parametrize("child_trust", [None, False, True])
def test_parent_component_does_not_authorize_nested_module(file_component, tmp_path, child_trust):
    child, marker = file_component
    if child_trust is not None:
        child["trusted"] = child_trust
    parent_path = tmp_path / "parent.py"
    parent_path.write_text(
        "from qlib.utils import init_instance_by_config\n"
        "class Parent:\n"
        "    def __init__(self, child):\n"
        "        self.child = init_instance_by_config(child)\n",
        encoding="utf-8",
    )
    config = {
        "class": "Parent",
        "module_path": str(parent_path),
        "trusted": True,
        "kwargs": {"child": child},
    }
    original = copy.deepcopy(config)
    if child_trust:
        assert init_instance_by_config(config).child.value == 42
        assert marker.exists()
    else:
        with pytest.raises(PermissionError, match="model.py"):
            init_instance_by_config(config)
        assert not marker.exists()
    assert config == original


@pytest.mark.parametrize("path_form", ["relative", "parent", "symlink", "home"])
def test_trusted_file_paths_preserve_resolution_not_containment(file_component, tmp_path, monkeypatch, path_form):
    config, marker = file_component
    monkeypatch.chdir(tmp_path)
    if path_form == "relative":
        path = "model.py"
    elif path_form == "parent":
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        monkeypatch.chdir(subdir)
        path = "../model.py"
    elif path_form == "home":
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        path = "~/model.py"
    else:
        link = tmp_path / "linked.py"
        try:
            link.symlink_to(config["module_path"])
        except (OSError, NotImplementedError):
            pytest.skip("Symlink creation is unavailable")
        path = str(link)
    with pytest.raises(PermissionError):
        get_module_by_module_path(path)
    assert not marker.exists()
    assert get_module_by_module_path(path, trusted=True).Model().value == 42


def test_trusted_module_still_requires_existing_python_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        get_module_by_module_path(str(tmp_path / "missing.py"), trusted=True)
    directory = tmp_path / "directory.py"
    directory.mkdir()
    with pytest.raises(ValueError, match="Python source file"):
        get_module_by_module_path(str(directory), trusted=True)


def test_failed_file_module_is_removed_from_module_cache(tmp_path):
    module_path = tmp_path / "broken_module.py"
    module_path.write_text("raise RuntimeError('broken')\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="broken"):
        get_module_by_module_path(str(module_path), trusted=True)
    assert not any(getattr(module, "__file__", None) == str(module_path) for module in list(sys.modules.values()))


def test_package_import_and_class_configs_do_not_require_file_trust():
    import qlib.data.base

    assert get_module_by_module_path("qlib.data.base") is qlib.data.base
    assert get_module_by_module_path(qlib.data.base) is qlib.data.base
    assert isinstance(
        init_instance_by_config({"class": "Feature", "module_path": "qlib.data.base", "kwargs": {"name": "close"}}),
        qlib.data.base.Feature,
    )
    assert init_instance_by_config({"class": dict, "kwargs": {"trusted": True}}) == {"trusted": True}
    with pytest.raises(TypeError, match="boolean"):
        init_instance_by_config({"class": dict, "trusted": "true"})


def test_register_wrapper_requires_its_own_consent(file_component):
    from qlib.utils import Wrapper, register_wrapper

    config, marker = file_component
    wrapper = Wrapper()
    with pytest.raises(PermissionError):
        register_wrapper(wrapper, "Model", config["module_path"])
    assert not marker.exists()
    register_wrapper(wrapper, "Model", config["module_path"], trusted=True)
    assert wrapper.value == 42


def test_custom_operators_use_per_config_consent(tmp_path, monkeypatch):
    from qlib.data.ops import Operators, register_all_ops
    from qlib.data.expression_parser import ExpressionSyntaxError, parse_expression

    monkeypatch.setattr(Operators, "_ops", Operators._ops.copy())
    module_path = tmp_path / "operators.py"
    marker = tmp_path / "operator-executed.txt"
    module_path.write_text(
        "from pathlib import Path\n"
        "from qlib.data.ops import Ref\n"
        f"Path({str(marker)!r}).write_text('executed')\n"
        "class FileRef(Ref): pass\n",
        encoding="utf-8",
    )
    operator = {"class": "FileRef", "module_path": str(module_path)}
    config = SimpleNamespace(custom_ops=[operator], trusted=True, trusted_module_roots=[str(tmp_path)])
    with pytest.raises(PermissionError):
        register_all_ops(config)
    assert not marker.exists()
    operator["trusted"] = True
    register_all_ops(config)
    assert str(parse_expression("FileRef($close, 1)")) == "FileRef($close,1)"
    with pytest.raises(ExpressionSyntaxError):
        parse_expression("__import__('os')")
    marker.unlink()
    del operator["trusted"]
    with pytest.raises(PermissionError):
        register_all_ops(config)
    assert not marker.exists()


def _instantiate_in_worker(config):
    return os.getpid(), init_instance_by_config(config).value


def test_saved_component_consent_survives_worker_loading_without_leaking(file_component):
    config, marker = file_component
    config["trusted"] = True
    restored = json.loads(json.dumps(config))
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as executor:
        pid, value = executor.submit(_instantiate_in_worker, restored).result(timeout=60)
        assert pid != os.getpid() and value == 42
        marker.unlink()
        restored.pop("trusted")
        with pytest.raises(PermissionError):
            executor.submit(_instantiate_in_worker, restored).result(timeout=60)
        assert not marker.exists()


def test_file_module_consent_does_not_authorize_pickle_loading(file_component, tmp_path):
    config, _ = file_component
    instance = init_instance_by_config(dict(config, trusted=True))
    artifact = tmp_path / "model.pkl"
    artifact.write_bytes(pickle.dumps(instance))
    with pytest.raises(pickle.UnpicklingError):
        init_instance_by_config(artifact, trusted=True)


def test_trusted_file_import_preserves_legacy_pickle_module_name(tmp_path, monkeypatch):
    module_path = tmp_path / "legacy_model.py"
    source = "class Model:\n    def __init__(self):\n        self.value = 42\n"
    module_path.write_text(source, encoding="utf-8")
    legacy_name = re.sub("^[^a-zA-Z_]+", "", re.sub("[^0-9a-zA-Z_]", "", str(module_path)[:-3].replace("/", "_")))
    old_module = ModuleType(legacy_name)
    exec(source, old_module.__dict__)
    monkeypatch.setitem(sys.modules, legacy_name, old_module)
    payload = pickle.dumps(old_module.Model())
    monkeypatch.delitem(sys.modules, legacy_name)
    with pytest.raises(PermissionError):
        get_module_by_module_path(str(module_path))
    assert legacy_name not in sys.modules
    module = get_module_by_module_path(str(module_path), trusted=True)
    loaded = pickle.loads(payload)  # Explicitly trusted artifact created above.
    assert isinstance(loaded, module.Model)
    assert loaded.value == 42


def test_legacy_alias_never_overwrites_an_existing_package(tmp_path, monkeypatch):
    existing = ModuleType("existing_package")
    monkeypatch.setitem(sys.modules, "existing_package", existing)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "existing_package.py").write_text("VALUE = 42\n", encoding="utf-8")
    module = get_module_by_module_path("existing_package.py", trusted=True)
    assert module.VALUE == 42
    assert sys.modules["existing_package"] is existing


@pytest.mark.parametrize("trusted", [False, True, "true"])
def test_tuner_file_import_uses_experiment_consent(tmp_path, trusted):
    from qlib.contrib.tuner.config import TunerConfigManager
    from qlib.contrib.tuner.pipeline import Pipeline
    from ruamel.yaml import YAML

    marker = tmp_path / "tuner-executed.txt"
    module_path = tmp_path / "tuner.py"
    module_path.write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('executed')\n"
        "class Tuner:\n"
        "    def __init__(self, config, optim):\n"
        "        self.config = config\n",
        encoding="utf-8",
    )
    config = {
        "experiment": {
            "tuner_module_path": str(module_path),
            "tuner_class": "Tuner",
            "trusted": trusted,
        },
    }
    path = tmp_path / "tuner.yaml"
    with path.open("w") as stream:
        YAML(typ="safe", pure=True).dump(config, stream)
    manager = TunerConfigManager(str(path))
    pipeline = SimpleNamespace(
        pipeline_ex_config=manager.pipeline_ex_config,
        qlib_client_config={},
        data_config={},
        backtest_config={},
        time_config={},
        optim_config=manager.optim_config,
    )
    assert manager.pipeline_ex_config.trusted == trusted
    if trusted is True:
        assert Pipeline.init_tuner(pipeline, 0, {"trainer": {}}).config["data"] == {}
        assert marker.exists()
    else:
        error = TypeError if isinstance(trusted, str) else PermissionError
        with pytest.raises(error):
            Pipeline.init_tuner(pipeline, 0, {"trainer": {}})
        assert not marker.exists()


@pytest.mark.parametrize(
    "relative_path, file_components",
    [
        ("TRA/configs/config_alstm.yaml", 2),
        ("TRA/configs/config_alstm_tra.yaml", 2),
        ("TRA/configs/config_alstm_tra_init.yaml", 2),
        ("TRA/configs/config_transformer.yaml", 2),
        ("TRA/configs/config_transformer_tra.yaml", 2),
        ("TRA/configs/config_transformer_tra_init.yaml", 2),
        ("LightGBM/workflow_config_lightgbm_multi_freq.yaml", 2),
        ("LightGBM/workflow_config_lightgbm_Alpha158_multi_freq.yaml", 1),
    ],
)
def test_file_based_examples_authorize_only_their_file_components(relative_path, file_components):
    from ruamel.yaml import YAML

    path = Path(__file__).resolve().parents[2] / "examples" / "benchmarks" / relative_path
    with path.open() as stream:
        config = YAML(typ="safe", pure=True).load(stream)
    assert "trusted" not in config["qlib_init"]
    assert "trusted_module_roots" not in config["qlib_init"]
    pending, seen, count = [config], set(), 0
    while pending:
        value = pending.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        if isinstance(value, dict):
            if str(value.get("module_path", "")).endswith(".py"):
                assert value["trusted"] is True
                assert "trusted" not in value.get("kwargs", {})
                count += 1
            elif "module_path" in value:
                assert "trusted" not in value
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
    assert count == file_components
