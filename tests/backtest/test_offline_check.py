"""100% offline tests for scripts/offline_check.py (no qlib package import).

Uses tmp_path fixtures only, so these tests pass in a bare interpreter with
just pytest + PyYAML + tomli installed.
"""

import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import offline_check  # noqa: E402 - path set up before import

PINNED = "actions/checkout@d23441a48e516b6c34aea4fa41551a30e30af803"


def _make_passing_repo(root: Path) -> Path:
    """Create a minimal synthetic repository that passes every check."""
    (root / ".github" / "workflows").mkdir(parents=True)
    (root / ".github" / "workflows" / "ci.yml").write_text(
        f"name: CI\njobs:\n  build:\n    steps:\n      - uses: {PINNED}\n", encoding="utf-8"
    )
    (root / "Makefile").write_text(
        "clean:\n\techo clean\n\n" + "".join(f"{t}:\n\techo {t}\n\n" for t in offline_check.REQUIRED_MAKE_TARGETS),
        encoding="utf-8",
    )
    (root / "pyproject.toml").write_text(
        "[tool.ruff]\nline-length = 120\n\n"
        "[tool.black]\nline-length = 120\n\n"
        "[tool.mypy]\n[[tool.mypy.overrides]]\nmodule = [\"qlib.x\"]\nstrict = true\n\n"
        "[tool.coverage.report]\nfail_under = 80\n\n"
        "[tool.pytest.ini_options]\naddopts = \"-q\"\n",
        encoding="utf-8",
    )
    (root / "qlib").mkdir()
    (root / "qlib" / "mod.py").write_text("def f():\n    try:\n        pass\n    except ValueError:\n        pass\n", encoding="utf-8")
    (root / "examples").mkdir()
    (root / "examples" / "demo.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "scripts").mkdir()
    (root / "scripts" / "tool.py").write_text("VALUE = 2\n", encoding="utf-8")
    (root / ".env.example").write_text("QLIB_REGION=cn\n", encoding="utf-8")
    (root / ".gitignore").write_text("__pycache__\n.env\n", encoding="utf-8")
    guide_dir = root / "docs" / "build"
    guide_dir.mkdir(parents=True)
    (guide_dir / "cross_platform_builds.rst").write_text(
        "Guide\n=====\n\nSection\n-------\n\nbody\n", encoding="utf-8"
    )
    (root / "docs" / "index.rst").write_text(
        "   Guide <build/cross_platform_builds.rst>\n", encoding="utf-8"
    )
    return root


def test_synthetic_repo_passes_all_checks(tmp_path) -> None:
    results = offline_check.run_all_checks(_make_passing_repo(tmp_path))
    assert {name: failures for name, failures in results.items() if failures} == {}


def test_unpinned_action_detected(tmp_path) -> None:
    root = _make_passing_repo(tmp_path)
    (root / ".github" / "workflows" / "ci.yml").write_text(
        "jobs:\n  build:\n    steps:\n      - uses: actions/checkout@v6\n", encoding="utf-8"
    )
    failures = offline_check.check_workflows(root)
    assert any("unpinned" in failure for failure in failures)


def test_bare_except_detected(tmp_path) -> None:
    root = _make_passing_repo(tmp_path)
    (root / "qlib" / "bad.py").write_text("try:\n    pass\nexcept:\n    pass\n", encoding="utf-8")
    failures = offline_check.check_security(root)
    assert any("bare except" in failure for failure in failures)


def test_fullloader_usage_detected(tmp_path) -> None:
    root = _make_passing_repo(tmp_path)
    (root / "scripts" / "loader.py").write_text(
        "import yaml\nC = yaml.load(S, Loader=yaml.FullLoader)\n", encoding="utf-8"
    )
    failures = offline_check.check_security(root)
    assert any("FullLoader" in failure for failure in failures)


def test_fullloader_mention_in_docstring_allowed(tmp_path) -> None:
    root = _make_passing_repo(tmp_path)
    (root / "scripts" / "noted.py").write_text(
        '"""Rejects ``yaml.FullLoader`` semantics (docs only)."""\n', encoding="utf-8"
    )
    assert offline_check.check_security(root) == []


def test_missing_env_ignore_detected(tmp_path) -> None:
    root = _make_passing_repo(tmp_path)
    (root / ".gitignore").write_text("__pycache__\n", encoding="utf-8")
    failures = offline_check.check_env_parity(root)
    assert any(".env" in failure for failure in failures)


def test_short_rst_underline_detected(tmp_path) -> None:
    root = _make_passing_repo(tmp_path)
    guide = root / "docs" / "build" / "cross_platform_builds.rst"
    guide.write_text("Long Section Title\n====\n\nbody\n", encoding="utf-8")
    failures = offline_check.check_guide_structure(root)
    assert any("underline" in failure for failure in failures)


def test_missing_makefile_target_detected(tmp_path) -> None:
    root = _make_passing_repo(tmp_path)
    (root / "Makefile").write_text("clean:\n\techo clean\n", encoding="utf-8")
    failures = offline_check.check_makefile(root)
    assert failures
    assert any("elevation-gates" in failure for failure in failures)


def test_missing_pyproject_section_detected(tmp_path) -> None:
    root = _make_passing_repo(tmp_path)
    (root / "pyproject.toml").write_text("[tool.ruff]\nline-length = 120\n", encoding="utf-8")
    failures = offline_check.check_pyproject(root)
    assert failures


def test_only_selection_runs_subset(tmp_path) -> None:
    results = offline_check.run_all_checks(_make_passing_repo(tmp_path), only=("workflows",))
    assert set(results) == {"workflows"}


def test_cli_exit_codes(tmp_path) -> None:
    root = _make_passing_repo(tmp_path)
    assert offline_check.main(["--root", str(root)]) == 0
    (root / ".env.example").unlink()
    assert offline_check.main(["--root", str(root)]) == 1


def test_makefile_lists_match_elevation_workflow() -> None:
    """Parity: the workflow must invoke make targets, never duplicate file lists."""
    text = (REPO_ROOT / ".github" / "workflows" / "engineering-elevation.yml").read_text(encoding="utf-8")
    for target in ("elevation-lint", "elevation-types", "elevation-gates"):
        assert f"make {target}" in text, f"workflow must call make {target}"
    # Long duplicated module lists belong in the Makefile variables only.
    assert "qlib/backtest/validation.py qlib/backtest/config_schema.py" not in text


def test_makefile_variables_are_internally_consistent() -> None:
    """Parity: every file in ELEVATION_FORMAT/TESTS/COV must exist on disk."""
    text = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    for variable in ("ELEVATION_FORMAT", "ELEVATION_TESTS"):
        match = re.search(rf"^{variable} := (.+)$", text, re.MULTILINE)
        assert match, f"{variable} missing from Makefile"
        for entry in match.group(1).split():
            assert (REPO_ROOT / entry).exists(), f"{variable} references missing file {entry}"


def test_real_repository_passes_offline() -> None:
    """Integration gate: the actual repo must satisfy its own policy."""
    results = offline_check.run_all_checks(REPO_ROOT)
    assert {name: failures for name, failures in results.items() if failures} == {}, results
