# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Post-build validation for cross-platform pyqlib wheels.

Checks every ``*.whl`` in ``--dist`` without installing the full dependency
stack (keeps validation fast and hermetic):

1. Filename parses as a wheel and tags a supported ``(py, abi, platform)`` triple
   (manylinux / macosx / win per arch, including ARM64).
2. The archive contains both compiled extensions
   (``rolling`` + ``expanding`` with the correct ``.so``/``.pyd`` suffix).
3. No platform-incompatible leakage (e.g. a ``.so`` inside a ``win_`` wheel).
4. ``METADATA`` declares the ``pyqlib`` name (guards misbuilt sdists).

Used by ``.github/workflows/build-wheels.yml`` and locally via
``python scripts/validate_wheels.py --dist dist``. Full numeric smoke tests
(import + rolling-op execution) run inside cibuildwheel's ``CIBW_TEST_COMMAND``
on the target runner; this script is the static gate that runs anywhere.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

# Allowed platform tags per artifact family. manylinux glibc 2.28+ supersedes the
# legacy manylinux2014 baseline; macOS floors match the build script
# (10.15 Intel / 11.0 Apple Silicon); Windows covers x64 + ARM64.
# Versioned manylinux tags (manylinux_2_28_*) match by glibc-floor prefix.
ALLOWED_MANYLINUX_GLIBC_FLOOR = (2, 28)
ALLOWED_PLATFORMS: dict[str, tuple[str, ...]] = {
    "linux": ("manylinux", "musllinux_"),
    "macos": ("macosx_10_15_x86_64", "macosx_11_0_arm64", "macosx_12_0_arm64", "macosx_universal2"),
    "windows": ("win_amd64", "win_arm64", "win32"),
}

REQUIRED_EXTENSIONS = ("rolling", "expanding")
NATIVE_SUFFIXES = (".so", ".pyd")


def _platform_family(platform_tag: str) -> str:
    if platform_tag.startswith(("manylinux", "musllinux")):
        return "linux"
    if platform_tag.startswith("macosx"):
        return "macos"
    if platform_tag.startswith("win"):
        return "windows"
    return "unknown"


def _manylinux_meets_floor(platform_tag: str) -> bool:
    """Check a ``manylinux_X_Y_arch`` tag against the glibc floor (2.28).

    Legacy ``manylinux2014``/``manylinux1`` tags (no X_Y version) predate the
    floor and are rejected.
    """
    try:
        rest = platform_tag.split("manylinux", 1)[1].lstrip("_")
        major, minor, _ = rest.split("_", 2)
        version = (int(major), int(minor))
    except (ValueError, IndexError, AttributeError):
        return False
    return version >= ALLOWED_MANYLINUX_GLIBC_FLOOR


def validate_wheel(path: Path) -> list[str]:
    """Return a list of validation errors (empty means the wheel is clean)."""
    errors: list[str] = []
    parts = path.name.split("-")
    if len(parts) < 5 or not path.suffix == ".whl":
        return [f"{path.name}: not a parseable wheel filename"]
    platform_tag = parts[-1].removesuffix(".whl")
    family = _platform_family(platform_tag)
    if family == "unknown":
        errors.append(f"{path.name}: unrecognized platform tag {platform_tag!r}")
        return errors
    if not any(platform_tag.startswith(allowed) for allowed in ALLOWED_PLATFORMS[family]):
        errors.append(f"{path.name}: platform tag {platform_tag!r} below the supported baseline")
    elif family == "linux" and platform_tag.startswith("manylinux"):
        if not _manylinux_meets_floor(platform_tag):
            errors.append(
                f"{path.name}: manylinux tag {platform_tag!r} below glibc "
                f"{ALLOWED_MANYLINUX_GLIBC_FLOOR[0]}.{ALLOWED_MANYLINUX_GLIBC_FLOOR[1]} baseline"
            )

    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
    except zipfile.BadZipFile:
        return [f"{path.name}: corrupt zip archive"]

    for ext in REQUIRED_EXTENSIONS:
        matches = [n for n in names if f"_libs/{ext}" in n and n.endswith(NATIVE_SUFFIXES)]
        if not matches:
            errors.append(f"{path.name}: missing compiled extension for {ext!r}")
    # Cross-family leakage: native binaries must match the wheel's OS.
    if family == "windows" and any(n.endswith(".so") for n in names if "_libs/" in n):
        errors.append(f"{path.name}: linux .so leaked into a Windows wheel")
    if family in ("linux", "macos") and any(n.endswith(".pyd") for n in names if "_libs/" in n):
        errors.append(f"{path.name}: windows .pyd leaked into a {family} wheel")
    metadata = next((n for n in names if n.endswith(".dist-info/METADATA")), None)
    if metadata is None:
        errors.append(f"{path.name}: missing dist-info METADATA")
    else:
        with zipfile.ZipFile(path) as archive:
            text = archive.read(metadata).decode("utf-8", errors="replace")
        if "Name: pyqlib" not in text:
            errors.append(f"{path.name}: METADATA does not declare pyqlib")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate built pyqlib wheels")
    parser.add_argument("--dist", default="dist", help="directory containing *.whl files")
    args = parser.parse_args(argv)

    wheels = sorted(Path(args.dist).glob("*.whl"))
    if not wheels:
        print(f"no wheels found in {args.dist}")
        return 2
    failures: list[str] = []
    for wheel in wheels:
        errors = validate_wheel(wheel)
        status = "OK" if not errors else "FAIL"
        print(f"[{status}] {wheel.name}")
        failures.extend(errors)
    for failure in failures:
        print(f"  - {failure}")
    print(f"validated {len(wheels)} wheel(s): {'clean' if not failures else f'{len(failures)} error(s)'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
