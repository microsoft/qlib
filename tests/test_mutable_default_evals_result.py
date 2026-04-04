# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Regression test for https://github.com/microsoft/qlib/issues/2167

Ensures that model.fit() methods use ``evals_result=None`` (with a guard)
instead of the mutable default ``evals_result=dict()``.  The latter is a
well-known Python antipattern that causes state leakage between calls.
"""

import ast
import os
import unittest
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent.parent / "qlib" / "contrib" / "model"
EXAMPLE_TRA = Path(__file__).resolve().parent.parent / "examples" / "benchmarks" / "TRA" / "src" / "model.py"


def _get_fit_defaults(filepath: Path):
    """Parse a Python file and return the default value node for evals_result
    in any fit() method found, along with the full source for body inspection."""
    source = filepath.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(source)
    results = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "fit":
            args = node.args
            # defaults are aligned to the end of args.args
            num_defaults = len(args.defaults)
            num_args = len(args.args)
            for i, default in enumerate(args.defaults):
                arg_index = num_args - num_defaults + i
                arg_name = args.args[arg_index].arg
                if arg_name == "evals_result":
                    results.append((node, default, source))
    return results


class TestNoMutableDefaultEvalsResult(unittest.TestCase):
    """Verify that no fit() method uses a mutable default for evals_result."""

    def _check_file(self, filepath: Path):
        fits = _get_fit_defaults(filepath)
        for func_node, default, source in fits:
            # The default must be None, not dict() or {}
            is_none = isinstance(default, ast.Constant) and default.value is None
            self.assertTrue(
                is_none,
                f"{filepath.name}:{func_node.lineno} - fit() uses mutable default "
                f"for evals_result (should be None)",
            )
            # The function body must contain the guard:
            #   if evals_result is None:
            #       evals_result = {}
            body_source = ast.get_source_segment(source, func_node)
            self.assertIn(
                "if evals_result is None:",
                body_source or "",
                f"{filepath.name}:{func_node.lineno} - fit() missing "
                f"if evals_result is None: evals_result = {{}} guard",
            )

    def test_all_contrib_models(self):
        """Check every .py file in qlib/contrib/model/."""
        py_files = sorted(MODEL_DIR.glob("*.py"))
        self.assertTrue(len(py_files) > 0, "No model files found")
        checked = 0
        for f in py_files:
            fits = _get_fit_defaults(f)
            if fits:
                self._check_file(f)
                checked += 1
        # Sanity: we expect at least 25 files to have evals_result in fit()
        self.assertGreaterEqual(checked, 25, "Too few model files checked")

    def test_example_tra_model(self):
        """Check the TRA example model too."""
        if EXAMPLE_TRA.exists():
            self._check_file(EXAMPLE_TRA)


if __name__ == "__main__":
    unittest.main()
