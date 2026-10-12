import ast
import importlib.util
from pathlib import Path


def test_adanos_feature_config_exposes_expected_fields():
    module_path = Path(__file__).resolve().parents[2] / "qlib" / "contrib" / "data" / "adanos_features.py"
    spec = importlib.util.spec_from_file_location("adanos_features", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    fields, names = module.get_adanos_feature_config()

    assert len(fields) == len(names)
    assert len(names) == len(set(names))
    assert "RETAIL_BUZZ_L1" in names
    assert "POLYMARKET_TRADES_RATIO5" in names
    assert any("$retail_buzz_avg" in field for field in fields)


def test_adanos_handler_extends_alpha158():
    module_path = Path(__file__).resolve().parents[2] / "qlib" / "contrib" / "data" / "handler_adanos.py"
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    class_def = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Alpha158AdanosUS")
    base_names = [base.id for base in class_def.bases if isinstance(base, ast.Name)]
    assert "Alpha158" in base_names
