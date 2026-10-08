# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""`scripts/collect_info.py` is what the bug report template asks users to run."""

import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "collect_info.py"

# Neither branch of the script may depend on what the machine happens to hold.
# setuptools stopped shipping pkg_resources with 82.0.0, so block the import. And
# `cython` is a build requirement that a plain install does not have, but a machine
# that does have it would skip the missing-package branch, so make one name missing.
MISSING = "cython"
DRIVER = textwrap.dedent(
    f"""
    import importlib.metadata
    import runpy
    import sys


    class BlockPkgResources:
        def find_spec(self, name, path=None, target=None):
            if name == "pkg_resources" or name.startswith("pkg_resources."):
                raise ImportError("pkg_resources is not available")
            return None


    sys.meta_path.insert(0, BlockPkgResources())
    sys.modules.pop("pkg_resources", None)

    _real_version = importlib.metadata.version


    def _version(name):
        if name == {MISSING!r}:
            raise importlib.metadata.PackageNotFoundError(name)
        return _real_version(name)


    importlib.metadata.version = _version
    sys.argv = ["collect_info.py", "all"]
    runpy.run_path(sys.argv[0], run_name="__main__")
    """
)


def run_collect_info():
    return subprocess.run(
        [sys.executable, "-c", DRIVER],
        cwd=SCRIPT.parent,
        capture_output=True,
        text=True,
        check=False,
    )


class TestCollectInfo(unittest.TestCase):
    def test_runs_without_pkg_resources(self):
        result = run_collect_info()
        self.assertEqual(result.returncode, 0, f"collect_info.py all failed:\n{result.stderr}")
        self.assertIn("Qlib version:", result.stdout)
        self.assertIn("numpy==", result.stdout)

    def test_reports_a_package_that_is_not_installed(self):
        result = run_collect_info()
        self.assertEqual(result.returncode, 0, f"collect_info.py all failed:\n{result.stderr}")
        self.assertIn(f"{MISSING}: not installed", result.stdout)


if __name__ == "__main__":
    unittest.main()
