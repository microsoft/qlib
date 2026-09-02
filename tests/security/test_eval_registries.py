import ast
from pathlib import Path


def _module_tree(relative_path):
    repository_root = Path(__file__).resolve().parents[2]
    return ast.parse((repository_root / relative_path).read_text(encoding="utf-8"))


def test_model_and_graph_dispatch_do_not_call_python_eval():
    for relative_path in [
        "qlib/contrib/model/pytorch_tra.py",
        "qlib/contrib/report/analysis_model/analysis_model_performance.py",
    ]:
        tree = _module_tree(relative_path)
        eval_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "eval"
        ]
        assert not eval_calls
