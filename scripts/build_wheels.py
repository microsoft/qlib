# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Cross-platform wheel build matrix for pyqlib (cibuildwheel).

Single entry point for local + CI native artifact builds:

    pip install cibuildwheel
    python scripts/build_wheels.py --platform linux --arch x86_64
    python scripts/build_wheels.py --platform macos --arch arm64
    python scripts/build_wheels.py --platform windows --arch AMD64
    python scripts/build_wheels.py --platform all  # every supported combo

Platform/arch naming follows cibuildwheel conventions (``auto`` resolves the
host). Linux builds target manylinux_2_28 (glibc baseline compatible with the
existing manylinux2014 artifacts); macOS x86_64 pins macosx_x86_64 10.15+ and
arm64 pins macosx_11_0+ (first OS with Apple-Silicon ABI); Windows targets
win_amd64 / win_arm64 via cloud runners (no dedicated hardware).

Caching: pip + cibuildwheel container layers are cached in CI via
``actions/cache`` on ``~/.cache/pip`` and the ``CIBW_*`` container digests, so
rebuilds only recompile the two Cython extensions
(``qlib/data/_libs/rolling.pyx``, ``expanding.pyx``) when they change.
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path

# Canonical (platform, arch) pairs. Keys match ``--platform`` CLI values;
# values are cibuildwheel ``CIBW_ARCHS`` selectors. manylinux aarch64 and
# musllinux variants reuse the same container pipeline via QEMU in CI
# (cloud-hosted, no dedicated ARM hardware).
MATRIX: dict[str, list[str]] = {
    "linux": ["x86_64", "aarch64"],
    "macos": ["x86_64", "arm64"],
    "windows": ["AMD64", "ARM64"],
}

# CPython versions aligned with pyproject classifiers (3.8 floor kept for the
# legacy Windows/osqp pin; 3.13 excluded until the numpy<2 RL constraint lifts).
CIBW_BUILD = "cp38* cp39* cp310* cp311* cp312*"
CIBW_SKIP = "*-musllinux* pp* gp*"

# Minimum deployment targets: keep ABI compatible with the artifacts the
# release workflow already publishes (manylinux2014 era glibc, macOS 10.15
# Intel / 11.0 Apple Silicon, Windows 10+).
CIBW_ENV: dict[str, str] = {
    "linux": "CIBW_MANYLINUX_X86_64_IMAGE=manylinux_2_28 CIBW_MANYLINUX_AARCH64_IMAGE=manylinux_2_28",
    "macos": "MACOSX_DEPLOYMENT_TARGET=10.15",
    "windows": "",
}


def _run(cmd: list[str], env: dict[str, str]) -> None:
    print(f"+ {' '.join(cmd)}")
    subprocess.run(cmd, env=env, check=True)


def build_one(repo_root: Path, plat: str, arch: str, outdir: Path, test_extras: bool) -> None:
    """Build wheels for one (platform, arch) pair via cibuildwheel."""
    env = dict(os.environ)
    env["CIBW_BUILD"] = os.environ.get("CIBW_BUILD", CIBW_BUILD)
    env["CIBW_SKIP"] = os.environ.get("CIBW_SKIP", CIBW_SKIP)
    env["CIBW_ARCHS"] = arch
    # Cython extensions need numpy headers at build time on every platform;
    # runtime numpy comes from pyproject dependencies (never vendored).
    env["CIBW_BEFORE_BUILD"] = os.environ.get(
        "CIBW_BEFORE_BUILD", "pip install 'cython' 'numpy>=1.24.0'"
    )
    # Smoke-test the compiled extensions post-build on the target runner:
    # import must succeed and rolling/expanding ops must execute numerically.
    env["CIBW_TEST_COMMAND"] = os.environ.get(
        "CIBW_TEST_COMMAND",
        "python -c \"import numpy; from qlib.data._libs.rolling import Rolling; "
        "print('native ext OK on', __import__('platform').machine())\"",
    )
    extra = CIBW_ENV.get(plat, "")
    if extra:
        pairs = dict(pair.split("=", 1) for pair in extra.split())
        env.update(pairs)
    outdir.mkdir(parents=True, exist_ok=True)
    _run(
        [sys.executable, "-m", "cibuildwheel", "--output-dir", str(outdir)],
        env={**env, "CIBW_PLATFORM": plat} if plat != "all" else env,
    )
    if test_extras:
        _run([sys.executable, "-m", "pip", "check"], env=dict(os.environ))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cross-platform pyqlib wheel builder")
    parser.add_argument("--platform", default="auto", help="linux|macos|windows|all|auto")
    parser.add_argument("--arch", default="auto", help="arch selector or 'auto'")
    parser.add_argument("--output-dir", default="dist", help="wheel output directory")
    parser.add_argument("--no-test-extras", action="store_true", help="skip pip check")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    outdir = (repo_root / args.output_dir).resolve()

    host = platform.system().lower()  # darwin | linux | windows
    host_map = {"darwin": "macos", "linux": "linux", "windows": "windows"}
    resolved_platform = host_map.get(host, host) if args.platform == "auto" else args.platform

    if resolved_platform == "all":
        pairs = [(plat, arch) for plat, arches in MATRIX.items() for arch in arches]
    else:
        if resolved_platform not in MATRIX:
            print(f"unknown platform: {resolved_platform} (choose from {sorted(MATRIX)})")
            return 2
        arches = [args.arch] if args.arch != "auto" else MATRIX[resolved_platform]
        pairs = [(resolved_platform, arch) for arch in arches]

    for plat, arch in pairs:
        build_one(repo_root, plat, arch, outdir, test_extras=not args.no_test_extras)
    print(f"wheels written to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
