import ast
from pathlib import Path
from unittest.mock import Mock

import pytest


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


def test_tra_rejects_code_as_model_name(tmp_path):
    pytest.importorskip("torch")
    from qlib.contrib.model.pytorch_tra import TRAModel

    marker = tmp_path / "executed.txt"
    model = object.__new__(TRAModel)
    model.logger = Mock()
    model.model_type = f"__import__('pathlib').Path({str(marker)!r}).touch()"
    with pytest.raises(ValueError, match="Unsupported model_type"):
        model._init_model()
    assert not marker.exists()


@pytest.mark.parametrize("model_type", ["RNN", "Transformer"])
def test_tra_initializes_supported_models(model_type):
    pytest.importorskip("torch")
    from qlib.contrib.model.pytorch_tra import MODEL_TYPES, TRAModel

    model = object.__new__(TRAModel)
    model.logger = Mock()
    model.model_type = model_type
    model.model_config = {"input_size": 6, "hidden_size": 8, "num_layers": 1}
    model.tra_config = {"num_states": 1}
    model.init_state = None
    model.reset_router = model.freeze_model = model.freeze_predictors = False
    model.lr = 0.001
    model._init_model()
    assert isinstance(model.model, MODEL_TYPES[model_type])


def test_graph_rejects_code_as_graph_name(tmp_path):
    pytest.importorskip("plotly")
    import pandas as pd
    from qlib.contrib.report.analysis_model.analysis_model_performance import model_performance_graph

    marker = tmp_path / "executed.txt"
    name = f"_import__('pathlib').Path({str(marker)!r}).touch()"
    with pytest.raises(ValueError, match="Unsupported graph name"):
        model_performance_graph(pd.DataFrame(), graph_names=[name], show_notebook=False)
    assert not marker.exists()


@pytest.mark.parametrize("name", ["group_return", "pred_ic", "pred_autocorr", "pred_turnover"])
def test_graph_dispatch_preserves_supported_names(monkeypatch, name):
    pytest.importorskip("plotly")
    import pandas as pd
    from qlib.contrib.report.analysis_model import analysis_model_performance as module

    figure = object()
    graph = Mock(return_value=(figure,))
    monkeypatch.setitem(module.GRAPH_FUNCTIONS, name, graph)
    result = module.model_performance_graph(pd.DataFrame(), graph_names=[name], show_notebook=False)
    assert result == [figure]
    graph.assert_called_once()
