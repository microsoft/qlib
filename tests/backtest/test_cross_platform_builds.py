"""Filesystem-free tests for cross-platform wheel validation logic."""

import zipfile

from scripts.validate_wheels import _platform_family, validate_wheel


def _make_wheel(tmp_path, filename, *, native_suffix=".so", name="pyqlib"):
    path = tmp_path / filename
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("qlib/data/_libs/rolling" + native_suffix, b"fake-native")
        archive.writestr("qlib/data/_libs/expanding" + native_suffix, b"fake-native")
        archive.writestr(
            f"{name}-0.0.0.dist-info/METADATA", f"Metadata-Version: 2.1\nName: {name}\n".encode()
        )
    return path


def test_valid_manylinux_wheel_passes(tmp_path) -> None:
    wheel = _make_wheel(tmp_path, "pyqlib-0.0.0-cp311-cp311-manylinux_2_28_x86_64.whl")
    assert validate_wheel(wheel) == []


def test_missing_extension_fails(tmp_path) -> None:
    path = tmp_path / "pyqlib-0.0.0-cp311-cp311-manylinux_2_28_x86_64.whl"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("qlib/data/_libs/rolling.so", b"fake-native")
        archive.writestr("pyqlib-0.0.0.dist-info/METADATA", b"Metadata-Version: 2.1\nName: pyqlib\n")
    errors = validate_wheel(path)
    assert any("expanding" in error for error in errors)


def test_windows_so_leakage_fails(tmp_path) -> None:
    wheel = _make_wheel(tmp_path, "pyqlib-0.0.0-cp311-cp311-win_amd64.whl", native_suffix=".so")
    errors = validate_wheel(wheel)
    assert any("leaked" in error for error in errors)


def test_wrong_package_name_fails(tmp_path) -> None:
    wheel = _make_wheel(
        tmp_path, "pyqlib-0.0.0-cp311-cp311-macosx_11_0_arm64.whl", native_suffix=".so", name="other"
    )
    errors = validate_wheel(wheel)
    assert any("METADATA" in error for error in errors)


def test_unknown_platform_tag_fails(tmp_path) -> None:
    wheel = _make_wheel(tmp_path, "pyqlib-0.0.0-cp311-cp311 risks-os.whl")
    errors = validate_wheel(wheel)
    assert errors


def test_platform_family_mapping() -> None:
    assert _platform_family("manylinux_2_28_x86_64") == "linux"
    assert _platform_family("musllinux_1_2_aarch64") == "linux"
    assert _platform_family("macosx_11_0_arm64") == "macos"
    assert _platform_family("win_arm64") == "windows"
    assert _platform_family("unknown_tag") == "unknown"


def test_old_manylinux_below_glibc_floor_fails(tmp_path) -> None:
    wheel = _make_wheel(tmp_path, "pyqlib-0.0.0-cp311-cp311-manylinux2014_x86_64.whl")
    errors = validate_wheel(wheel)
    assert any("glibc" in error for error in errors)
