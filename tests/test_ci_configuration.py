# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Offline regression checks for the shared source/PyPI CI compatibility policy."""

import ast
import json
from pathlib import Path
import shlex
import unittest

from packaging.markers import default_environment
from packaging.requirements import Requirement
import yaml

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = (
    "test_qlib_from_source.yml",
    "test_qlib_from_source_slow.yml",
    "test_qlib_from_pip.yml",
)


class TestCIConfiguration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workflows = {
            name: yaml.safe_load((ROOT / ".github/workflows" / name).read_text(encoding="utf-8")) for name in WORKFLOWS
        }
        cls.constraints = {
            requirement.name: requirement
            for line in (ROOT / ".github/ci/constraints.txt").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
            for requirement in [Requirement(line)]
        }

    def test_constraints_match_package_metadata(self):
        project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
        requirements = list(project["dependencies"])
        for extra in project["optional-dependencies"].values():
            requirements.extend(extra)
        requirements = {req.name: req for req in map(Requirement, requirements)}
        for name, constraint in self.constraints.items():
            with self.subTest(package=name):
                self.assertEqual(constraint.specifier, requirements[name].specifier)
                self.assertEqual(constraint.marker, requirements[name].marker)

    def test_known_bad_versions_are_excluded(self):
        for name, version in (
            ("mlflow", "3.15.1"),
            ("filelock", "3.30.0"),
            ("fastjsonschema", "2.22.0"),
            ("plotly", "7.0.0"),
            ("lxml", "6.1.3"),
            ("osqp", "1.1.3"),
            ("osqp", "0.6.7.post3"),
            ("black", "26.1.0"),
        ):
            with self.subTest(package=name, version=version):
                self.assertNotIn(version, self.constraints[name].specifier)

    def test_compatible_versions_are_allowed(self):
        for name, version in (
            ("mlflow", "2.17.2"),
            ("mlflow", "3.12.0"),
            ("filelock", "3.16.1"),
            ("fastjsonschema", "2.21.2"),
            ("plotly", "6.9.0"),
            ("lxml", "6.1.2"),
            ("osqp", "1.0.5"),
            ("black", "25.12.0"),
        ):
            with self.subTest(package=name, version=version):
                self.assertIn(version, self.constraints[name].specifier)

    def test_platform_bounds_do_not_leak_to_other_environments(self):
        for platform in ("win32", "linux", "darwin"):
            for version in ("3.8", "3.9", "3.10", "3.11", "3.12"):
                env = dict(default_environment(), sys_platform=platform, python_version=version)
                with self.subTest(platform=platform, python=version):
                    self.assertEqual(
                        self.constraints["osqp"].marker.evaluate(env), platform == "win32" and version == "3.8"
                    )
                    self.assertEqual(self.constraints["fastjsonschema"].marker.evaluate(env), version in ("3.8", "3.9"))

    def test_all_pip_subprocesses_inherit_absolute_constraints(self):
        for name, workflow in self.workflows.items():
            with self.subTest(workflow=name):
                job = workflow["jobs"]["build"]
                self.assertEqual(job["env"]["PIP_CONSTRAINT"], "${{ github.workspace }}/.github/ci/constraints.txt")
                self.assertEqual(job["env"]["MLFLOW_ALLOW_FILE_STORE"], "true")
                for step in job["steps"]:
                    self.assertNotIn("PIP_CONSTRAINT", step.get("env", {}))

    def test_full_matrix_is_retained_and_failures_are_not_hidden(self):
        for name, workflow in self.workflows.items():
            with self.subTest(workflow=name):
                job = workflow["jobs"]["build"]
                self.assertFalse(job["strategy"]["fail-fast"])
                matrix = job["strategy"]["matrix"]
                self.assertEqual(set(matrix["python-version"]), {"3.8", "3.9", "3.10", "3.11", "3.12"})
                self.assertEqual(
                    set(matrix["os"]), {"windows-latest", "ubuntu-24.04", "ubuntu-22.04", "macos-14", "macos-15"}
                )
                self.assertNotIn("exclude", matrix)
                self.assertNotIn("continue-on-error", job)
                for step in job["steps"]:
                    self.assertNotIn("continue-on-error", step)
                    self.assertNotIn("continue_on_error", step.get("with", {}))

    def test_grpcio_source_build_is_scoped_and_precedes_installation(self):
        condition = (
            "${{ runner.os == 'macOS' && runner.arch == 'ARM64' && "
            "(matrix.python-version == '3.8' || matrix.python-version == '3.9') }}"
        )
        for name in ("test_qlib_from_source.yml", "test_qlib_from_source_slow.yml"):
            with self.subTest(workflow=name):
                steps = self.workflows[name]["jobs"]["build"]["steps"]
                configure = next(step for step in steps if step["name"].startswith("Configure grpcio source builds"))
                install = next(step for step in steps if step["name"] == "Set up Python tools")
                native = next(step for step in steps if step["name"].startswith("Verify grpcio native extension"))
                check = next(step for step in steps if step["name"] == "Verify dependency consistency")
                self.assertEqual(configure["if"], condition)
                self.assertEqual(native["if"], condition)
                self.assertEqual(configure["shell"], "bash")
                self.assertIn('echo "PIP_NO_BINARY=grpcio" >> "$GITHUB_ENV"', configure["run"])
                self.assertIn('echo "GRPC_PYTHON_BUILD_EXT_COMPILER_JOBS=2" >> "$GITHUB_ENV"', configure["run"])
                self.assertIn("from grpc._cython import cygrpc", native["run"])
                self.assertLess(steps.index(configure), steps.index(install))
                self.assertLess(steps.index(install), steps.index(native))
                self.assertLess(steps.index(native), steps.index(check))
                for step in steps:
                    if "pip install" in step.get("run", ""):
                        self.assertLess(steps.index(configure), steps.index(step))
                # Later installs must not restore the broken binary wheel.
                self.assertNotIn("PIP_NO_BINARY", self.workflows[name]["jobs"]["build"].get("env", {}))
                for step in steps:
                    self.assertNotIn("PIP_NO_BINARY", step.get("env", {}))

    def test_download_retries_are_bounded_and_noninteractive(self):
        for name, workflow in self.workflows.items():
            downloads = [
                step
                for step in workflow["jobs"]["build"]["steps"]
                if "data download" in step["name"].lower() or "downloads dependencies" in step["name"].lower()
            ]
            with self.subTest(workflow=name):
                self.assertTrue(downloads)
                for step in downloads:
                    options = step["with"]
                    self.assertEqual(options["max_attempts"], 3)
                    self.assertEqual(options["timeout_minutes"], 15)
                    self.assertEqual(options["shell"], "bash")
                    commands = options["command"]
                    for line in commands.splitlines():
                        if "qlib_data --" in line or "download_data --" in line:
                            self.assertIn("--delete_old False", line)
                            args = shlex.split(line)
                            self.assertIs(ast.literal_eval(args[args.index("--delete_old") + 1]), False)
                    if len(commands.splitlines()) > 1:
                        self.assertIn("set -euo pipefail", commands)

    def test_pypi_workflow_does_not_import_the_checkout(self):
        steps = self.workflows["test_qlib_from_pip.yml"]["jobs"]["build"]["steps"]
        for name in ("Downloads dependencies data", "Test workflow by config"):
            command = next(step for step in steps if step["name"] == name)["with"]["command"]
            self.assertIn('cd "$RUNNER_TEMP"', command)
            self.assertLess(command.index('cd "$RUNNER_TEMP"'), command.index("python"))
        command = next(step for step in steps if step["name"] == "Test workflow by config")["with"]["command"]
        self.assertIn('"$GITHUB_WORKSPACE/examples/benchmarks/LightGBM/', command)
        self.assertIn("not in pathlib.Path(qlib.__file__).resolve().parents", command)

    def test_consistency_and_native_import_are_checked(self):
        for name, workflow in self.workflows.items():
            with self.subTest(workflow=name):
                steps = workflow["jobs"]["build"]["steps"]
                self.assertTrue(any(step.get("run") == "python -m pip check" for step in steps))
                native = next(step for step in steps if step["name"] == "Verify native solvers on Windows Python 3.8")
                self.assertIn("matrix.python-version == '3.8'", native["if"])
                self.assertIn("matrix.os == 'windows-latest'", native["if"])
                self.assertIn("import cvxpy, osqp", native["run"])
                self.assertEqual(steps[-1]["name"], "Report installed dependencies")
                self.assertEqual(steps[-1]["if"], "always()")

    def test_title_lint_uses_a_lockfile_and_no_dynamic_download(self):
        workflow = yaml.safe_load((ROOT / ".github/workflows/lint_title.yml").read_text(encoding="utf-8"))
        steps = workflow["jobs"]["lint-title"]["steps"]
        setup = next(step for step in steps if step["name"] == "Setup Node.js")
        self.assertEqual(setup["with"]["node-version"], "22")
        install = next(step for step in steps if step["name"] == "Install commitlint")
        self.assertIn("npm ci", install["run"])
        validate = next(step for step in steps if step["name"] == "Validate PR Title with commitlint")
        self.assertIn("npx --no-install", validate["run"])
        self.assertEqual(validate["env"]["BODY"], "${{ github.event.pull_request.title }}")
        self.assertNotIn("${{", validate["run"])
        manifest = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))
        lock = json.loads((ROOT / "package-lock.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["devDependencies"], lock["packages"][""]["devDependencies"])
        for name, version in manifest["devDependencies"].items():
            self.assertEqual(version, lock["packages"]["node_modules/" + name]["version"])


if __name__ == "__main__":
    unittest.main()
