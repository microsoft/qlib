# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Offline repository conformance checker for the engineering-elevation policy.

Runs 100% offline with the standard library plus PyYAML and tomli (already
present in every dev/CI environment). No network, no subprocess, no imports
from the ``qlib`` package itself, so it works before dependencies are
installed. Use it as a pre-push sanity gate or a reviewer aid::

    python scripts/offline_check.py            # check this repository
    python scripts/offline_check.py --root DIR # check a staged/alternate tree

Checks (each returns a list of human-readable failures; empty == pass):

1. ``workflows``       - every ``.github/workflows/*.yml`` parses as YAML and
                         every remote ``uses:`` reference is pinned to a full
                         40-character commit SHA.
2. ``makefile``        - required elevation/packaging targets are declared.
3. ``pyproject``       - ``tool.ruff``, ``tool.black``, ``tool.mypy`` (with
                         non-empty strict overrides), ``tool.coverage`` with
                         ``fail_under``, and ``tool.pytest.ini_options`` exist.
4. ``security``        - no bare ``except:`` in ``qlib/`` and no
                         ``Loader=yaml.FullLoader`` usage anywhere in source,
                         examples, or scripts.
5. ``env_parity``      - ``.env.example`` exists and ``.gitignore`` ignores
                         ``.env`` (local/container parity contract).
6. ``guide_structure`` - the cross-platform build guide exists, its section
                         underlines are valid RST, and ``docs/index.rst``
                         references it.

Exit codes: ``0`` all checks pass, ``1`` one or more failures, ``2`` bad CLI.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Callable, Dict, List

import yaml

try:  # Python 3.11+ stdlib; tomli backport is pinned for older interpreters
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on py<3.11 CI legs
    import tomli as tomllib

# Full-SHA pin policy, identical to tests/test_ci_configuration.py.
_SHA_PIN_RE = re.compile(r"^[\w.-]+/[\w./-]+@[0-9a-f]{40}$")
# Bare-except sweep (the pattern the elevation brief prohibits).
_BARE_EXCEPT_RE = re.compile(r"^\s*except\s*:", re.MULTILINE)
# Usage form only: doc mentions of FullLoader are allowed, code is not.
_FULL_LOADER_RE = re.compile(r"Loader\s*=\s*yaml\.FullLoader")

REQUIRED_MAKE_TARGETS = (
    "elevation-gates",
    "elevation-lint",
    "elevation-types",
    "elevation-tests",
    "schema-check",
    "env-check",
    "offline-check",
    "wheels",
    "sdist",
)

REQUIRED_PYPROJECT_SECTIONS = ("ruff", "black", "mypy", "coverage", "pytest")

# The checker documents the forbidden patterns verbatim in its own docstring;
# exclude it from the security sweep (a linter never trips on its own rule).
_SELF_EXCLUDED = frozenset({"scripts/offline_check.py"})


def check_workflows(root: Path) -> List[str]:
    """Validate workflow YAML syntax and full-SHA action pins."""
    failures: List[str] = []
    workflow_dir = root / ".github" / "workflows"
    if not workflow_dir.is_dir():
        return [f"workflows: missing directory {workflow_dir}"]
    for path in sorted(workflow_dir.glob("*.yml")):
        try:
            workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            failures.append(f"workflows: {path.name} is not valid YAML: {exc}")
            continue
        for job in (workflow or {}).get("jobs", {}).values():
            for entry in [job] + list(job.get("steps", [])):
                reference = entry.get("uses", "")
                if reference and not reference.startswith("./") and not _SHA_PIN_RE.match(reference):
                    failures.append(f"workflows: {path.name} has unpinned action {reference!r}")
    return failures


def check_makefile(root: Path) -> List[str]:
    """Validate that required elevation/packaging targets are declared."""
    makefile = root / "Makefile"
    if not makefile.is_file():
        return ["makefile: Makefile is missing"]
    declared = {
        match.group(1)
        for line in makefile.read_text(encoding="utf-8").splitlines()
        for match in [re.match(r"^([A-Za-z0-9_-]+):", line)]
        if match
    }
    missing = [target for target in REQUIRED_MAKE_TARGETS if target not in declared]
    return [f"makefile: missing target {target!r}" for target in missing]


def check_pyproject(root: Path) -> List[str]:
    """Validate tool configuration sections in pyproject.toml."""
    pyproject = root / "pyproject.toml"
    if not pyproject.is_file():
        return ["pyproject: pyproject.toml is missing"]
    document = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    tool = document.get("tool", {})
    failures = [
        f"pyproject: missing [tool.{section}]"
        for section in REQUIRED_PYPROJECT_SECTIONS
        if section not in tool
    ]
    overrides = tool.get("mypy", {}).get("overrides", [])
    if not overrides or not any(entry.get("strict") for entry in overrides):
        failures.append("pyproject: mypy overrides must declare at least one strict module set")
    if tool.get("coverage", {}).get("report", {}).get("fail_under") is None:
        failures.append("pyproject: coverage.report.fail_under is not set")
    return failures


def check_security(root: Path) -> List[str]:
    """Sweep source trees for bare excepts and unsafe FullLoader usage."""
    failures: List[str] = []
    for tree in ("qlib", "examples", "scripts"):
        base = root / tree
        if not base.is_dir():
            failures.append(f"security: missing directory {base}")
            continue
        for path in sorted(base.rglob("*.py")):
            relative = path.relative_to(root).as_posix()
            if relative in _SELF_EXCLUDED:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            if tree == "qlib" and _BARE_EXCEPT_RE.search(text):
                failures.append(f"security: bare except: in {relative}")
            if _FULL_LOADER_RE.search(text):
                failures.append(f"security: yaml.FullLoader usage in {relative}")
    return failures


def check_env_parity(root: Path) -> List[str]:
    """Validate the .env.example / .gitignore local-container parity contract."""
    failures: List[str] = []
    if not (root / ".env.example").is_file():
        failures.append("env_parity: .env.example is missing")
    gitignore = root / ".gitignore"
    if not gitignore.is_file():
        failures.append("env_parity: .gitignore is missing")
    elif ".env" not in gitignore.read_text(encoding="utf-8").splitlines():
        failures.append("env_parity: .gitignore does not ignore .env")
    return failures


def check_guide_structure(root: Path) -> List[str]:
    """Validate RST guide structure and its docs/index.rst registration."""
    failures: List[str] = []
    guide = root / "docs" / "build" / "cross_platform_builds.rst"
    if not guide.is_file():
        return ["guide_structure: docs/build/cross_platform_builds.rst is missing"]
    lines = guide.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if line and set(line) in ({"="}, {"-"}, {"~"}) and index > 0:
            title = lines[index - 1]
            if title and len(line) < len(title):
                failures.append(f"guide_structure: underline too short for title {title!r}")
    index_rst = root / "docs" / "index.rst"
    if not index_rst.is_file():
        failures.append("guide_structure: docs/index.rst is missing")
    elif "build/cross_platform_builds.rst" not in index_rst.read_text(encoding="utf-8"):
        failures.append("guide_structure: docs/index.rst does not reference the build guide")
    return failures


CHECKS: Dict[str, Callable[[Path], List[str]]] = {
    "workflows": check_workflows,
    "makefile": check_makefile,
    "pyproject": check_pyproject,
    "security": check_security,
    "env_parity": check_env_parity,
    "guide_structure": check_guide_structure,
}


def run_all_checks(root: Path, only: tuple = ()) -> Dict[str, List[str]]:
    """Run every (or selected) check against ``root``; returns name -> failures."""
    selected = {name: CHECKS[name] for name in only} if only else CHECKS
    return {name: check(root) for name, check in selected.items()}


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline repository conformance checker")
    parser.add_argument("--root", default=".", help="repository root to check")
    parser.add_argument(
        "--only",
        nargs="*",
        choices=sorted(CHECKS),
        default=(),
        help="run only the named checks",
    )
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    results = run_all_checks(root, only=tuple(args.only))
    failures = 0
    for name, messages in results.items():
        status = "OK" if not messages else "FAIL"
        print(f"[{status}] {name}")
        for message in messages:
            print(f"  - {message}")
            failures += 1
    print(f"offline check: {len(results)} checks, {failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
