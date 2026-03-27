"""Test for Issue #1890: Model result state leakage between sequential fits.

The root cause was a mutable default argument `evals_result=dict()` in
TRAModel.fit(), which is a classic Python antipattern where the same dict
object is shared across all calls that use the default.
"""
import ast
import os
import pytest


class TestMutableDefaultArgs:
    """Verify that mutable default arguments are not used in model fit/predict."""

    def _get_default_args(self, filepath, method_name):
        """Parse a Python file's AST and return default values for a method."""
        with open(filepath) as f:
            tree = ast.parse(f.read())

        results = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == method_name:
                    # Collect defaults for the function
                    defaults = node.args.defaults
                    for d in defaults:
                        results.append(ast.dump(d))
        return results

    def test_tra_fit_no_mutable_default(self):
        """TRAModel.fit should use None, not dict(), as the default for evals_result."""
        tra_path = os.path.join(
            os.path.dirname(__file__), "..", "qlib", "contrib", "model", "pytorch_tra.py"
        )
        with open(tra_path) as f:
            tree = ast.parse(f.read())

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "fit":
                # Check the last default argument (evals_result)
                for default in node.args.defaults:
                    # Should NOT be Call(func=Name(id='dict')) i.e. dict()
                    if isinstance(default, ast.Call):
                        if isinstance(default.func, ast.Name) and default.func.id == "dict":
                            pytest.fail(
                                "TRAModel.fit still has mutable default `evals_result=dict()`. "
                                "Should be `evals_result=None`."
                            )
                    # Should NOT be Dict (i.e. {})
                    if isinstance(default, ast.Dict) and len(default.keys) == 0:
                        pytest.fail(
                            "TRAModel.fit still has mutable default `evals_result={}`. "
                            "Should be `evals_result=None`."
                        )

    def test_dict_default_shared_across_calls(self):
        """Demonstrate the mutable default antipattern is fixed.

        With `def f(x=dict())`, calling f() twice returns the SAME dict object.
        After fix with `def f(x=None)`, each call gets a fresh dict.
        """
        # Simulate the fixed pattern
        def fixed_func(evals_result=None):
            if evals_result is None:
                evals_result = {}
            evals_result["key"] = "value"
            return evals_result

        result1 = fixed_func()
        result2 = fixed_func()
        # Each call should get an independent dict
        assert result1 is not result2

    def test_mutable_default_antipattern(self):
        """Demonstrate why dict() default is dangerous."""
        # This is the BAD pattern (for demonstration only)
        def bad_func(evals_result={}):  # noqa: B006
            evals_result["data"] = evals_result.get("data", [])
            evals_result["data"].append(1)
            return evals_result

        r1 = bad_func()
        r2 = bad_func()
        # r1 and r2 are the SAME object — state leaked!
        assert r1 is r2  # This proves the antipattern leaks state
        assert len(r1["data"]) == 2  # Two appends to the same list

    def test_no_mutable_defaults_in_model_fit_predict(self):
        """Scan all contrib model files for mutable defaults in fit/predict via AST."""
        model_dir = os.path.join(
            os.path.dirname(__file__), "..", "qlib", "contrib", "model"
        )
        mutable_defaults = []

        for fname in os.listdir(model_dir):
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(model_dir, fname)
            try:
                with open(fpath) as f:
                    tree = ast.parse(f.read())
            except SyntaxError:
                continue

            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if node.name not in ("fit", "predict"):
                    continue

                # Check defaults
                arg_names = [a.arg for a in node.args.args]
                # defaults align to the last N args
                n_defaults = len(node.args.defaults)
                defaulted_args = arg_names[-n_defaults:] if n_defaults else []

                for arg_name, default in zip(defaulted_args, node.args.defaults):
                    is_mutable = False
                    if isinstance(default, (ast.Dict, ast.List, ast.Set)):
                        is_mutable = True
                    elif isinstance(default, ast.Call):
                        func = default.func
                        if isinstance(func, ast.Name) and func.id in ("dict", "list", "set"):
                            is_mutable = True
                    if is_mutable:
                        mutable_defaults.append(
                            f"{fname}:{node.name}({arg_name}=<mutable>) @ line {node.lineno}"
                        )

        assert mutable_defaults == [], (
            f"Mutable default arguments found in model fit/predict methods: {mutable_defaults}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
