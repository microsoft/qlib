import sys

import pytest

from qlib.utils.mod import get_module_by_module_path, init_instance_by_config, set_trusted_module_roots


@pytest.fixture(autouse=True)
def isolate_trusted_roots(monkeypatch):
    monkeypatch.setattr("qlib.utils.mod._TRUSTED_MODULE_ROOTS", [])


def test_file_module_loading_is_disabled_by_default(tmp_path):
    module_path = tmp_path / "custom_module.py"
    module_path.write_text("VALUE = 1\n", encoding="utf-8")

    with pytest.raises(PermissionError, match="disabled by default"):
        get_module_by_module_path(str(module_path))


def test_file_module_loading_accepts_trusted_root(tmp_path):
    module_path = tmp_path / "custom_module.py"
    module_path.write_text("VALUE = 1\n", encoding="utf-8")

    module = get_module_by_module_path(str(module_path), allowed_module_roots=[tmp_path])

    assert module.VALUE == 1


def test_file_module_loading_uses_configured_trusted_roots(tmp_path):
    module_path = tmp_path / "custom_module.py"
    module_path.write_text("VALUE = 1\n", encoding="utf-8")
    set_trusted_module_roots([tmp_path])
    try:
        module = get_module_by_module_path(str(module_path))
    finally:
        set_trusted_module_roots([])

    assert module.VALUE == 1


def test_file_module_loading_rejects_path_outside_trusted_root(tmp_path):
    trusted_root = tmp_path / "trusted"
    trusted_root.mkdir()
    module_path = tmp_path / "outside.py"
    module_path.write_text("VALUE = 1\n", encoding="utf-8")

    with pytest.raises(PermissionError, match="outside the allowed module roots"):
        get_module_by_module_path(str(module_path), allowed_module_roots=[trusted_root])


def test_failed_file_module_is_removed_from_module_cache(tmp_path):
    module_path = tmp_path / "broken_module.py"
    module_path.write_text("raise RuntimeError('broken')\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="broken"):
        get_module_by_module_path(str(module_path), allowed_module_roots=[tmp_path])

    assert not any(getattr(module, "__file__", None) == str(module_path) for module in sys.modules.values())


@pytest.mark.parametrize("escape", ["parent", "symlink"])
def test_config_module_rejects_escape_before_execution(tmp_path, escape):
    trusted_root = tmp_path / "trusted"
    trusted_root.mkdir()
    marker = tmp_path / "executed.txt"
    outside = tmp_path / "outside.py"
    outside.write_text(f"open({str(marker)!r}, 'w').write('executed')\nclass Model: pass\n", encoding="utf-8")
    if escape == "parent":
        module_path = trusted_root / ".." / outside.name
    else:
        module_path = trusted_root / "linked.py"
        try:
            module_path.symlink_to(outside)
        except (OSError, NotImplementedError):
            pytest.skip("Symlink creation is unavailable")
    config = {"class": "Model", "module_path": str(module_path)}
    with pytest.raises(PermissionError, match="outside the allowed module roots"):
        init_instance_by_config(config, allowed_module_roots=[trusted_root])
    assert not marker.exists()


def test_config_module_accepts_explicit_trust_and_constructor_arguments(tmp_path):
    module_path = tmp_path / "model.py"
    module_path.write_text(
        "class Model:\n    def __init__(self, value):\n        self.value = value\n", encoding="utf-8"
    )
    config = {"class": "Model", "module_path": str(module_path), "kwargs": {"value": 42}}
    assert init_instance_by_config(config, allowed_module_roots=[tmp_path]).value == 42


def test_empty_explicit_roots_override_process_trust(tmp_path):
    module_path = tmp_path / "model.py"
    module_path.write_text("VALUE = 1\n", encoding="utf-8")
    set_trusted_module_roots([tmp_path])
    with pytest.raises(PermissionError, match="disabled by default"):
        get_module_by_module_path(str(module_path), allowed_module_roots=[])


def test_scalar_root_cannot_accidentally_trust_filesystem_root(tmp_path):
    module_path = tmp_path / "model.py"
    module_path.write_text("raise RuntimeError('must not execute')\n", encoding="utf-8")
    with pytest.raises(TypeError, match="sequence"):
        get_module_by_module_path(str(module_path), allowed_module_roots=str(tmp_path))
    with pytest.raises(TypeError, match="sequence"):
        set_trusted_module_roots(str(tmp_path))


def test_package_import_does_not_require_file_trust():
    import qlib.data.base

    assert get_module_by_module_path("qlib.data.base") is qlib.data.base
