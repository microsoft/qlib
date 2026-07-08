import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]


def load_fxmacrodata_collector():
    fire_module = types.ModuleType("fire")
    fire_module.Fire = lambda *_args, **_kwargs: None

    base_module = types.ModuleType("data_collector.base")

    class BaseCollector:
        INTERVAL_1d = "1d"

    class BaseNormalize:
        def __init__(self, date_field_name="date", symbol_field_name="symbol", **kwargs):
            self._date_field_name = date_field_name
            self._symbol_field_name = symbol_field_name
            self.kwargs = kwargs

    class BaseRun:
        pass

    class Normalize:
        pass

    base_module.BaseCollector = BaseCollector
    base_module.BaseNormalize = BaseNormalize
    base_module.BaseRun = BaseRun
    base_module.Normalize = Normalize

    package_module = types.ModuleType("data_collector")
    package_module.__path__ = [str(ROOT_DIR / "scripts" / "data_collector")]
    stubs = {
        "fire": fire_module,
        "data_collector": package_module,
        "data_collector.base": base_module,
    }
    collector_path = ROOT_DIR / "scripts" / "data_collector" / "fxmacrodata" / "collector.py"
    spec = importlib.util.spec_from_file_location("fxmacrodata_collector", collector_path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module


def test_macro_rows_to_frame_flattens_announcement_and_prediction_data():
    collector = load_fxmacrodata_collector()

    data = collector.FXMacroDataMacroCollector._rows_to_macro_frame(
        "usd",
        "inflation",
        [
            {
                "date": "2026-05-31",
                "val": 4.2,
                "announcement_datetime": 1781094600,
                "consensus": 3.9,
                "forecast": 4.0,
                "surprise": 0.3,
                "predictions": [{"predicted_value": 4.1}],
            }
        ],
    )

    assert list(data.columns) == collector.MACRO_OUTPUT_COLUMNS
    assert data.loc[0, "symbol"] == "usd_inflation"
    assert data.loc[0, "value"] == 4.2
    assert data.loc[0, "consensus"] == 3.9
    assert data.loc[0, "prediction"] == 4.1
    assert data.loc[0, "prediction_count"] == 1.0
    assert data.loc[0, "announcement_datetime"] == 1781094600


def test_macro_normalize_keeps_numeric_feature_columns():
    collector = load_fxmacrodata_collector()
    normalizer = collector.FXMacroDataMacroNormalize()

    data = normalizer.normalize(
        pd.DataFrame(
            {
                "date": ["2026-05-31"],
                "symbol": ["USD_INFLATION"],
                "value": ["4.2"],
                "actual": ["4.2"],
                "prediction": ["4.1"],
                "announcement_datetime": ["1781094600"],
            }
        )
    )

    assert data.loc[0, "date"] == "2026-05-31"
    assert data.loc[0, "symbol"] == "usd_inflation"
    assert data.loc[0, "value"] == 4.2
    assert data.loc[0, "prediction"] == 4.1
    assert data.loc[0, "announcement_datetime"] == 1781094600
