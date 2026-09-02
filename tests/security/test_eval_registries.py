import pandas as pd
import pytest

pytest.importorskip("plotly")

from qlib.contrib.report.analysis_model.analysis_model_performance import model_performance_graph


def test_model_performance_rejects_unknown_graph_name():
    with pytest.raises(ValueError, match="Unsupported graph name"):
        model_performance_graph(pd.DataFrame(), graph_names=["__import__"], show_notebook=False)
