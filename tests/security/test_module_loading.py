import sys

import pytest

from qlib.utils.mod import get_module_by_module_path, set_trusted_module_roots


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
