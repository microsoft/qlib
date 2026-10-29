"""Regression tests for the standalone data-health-checker script."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest


@pytest.fixture
def checker_class(monkeypatch):
    # CSV checks do not need a configured Qlib provider or market data.
    qlib_stub = ModuleType("qlib")
    qlib_data_stub = ModuleType("qlib.data")
    qlib_data_stub.D = object()
    monkeypatch.setitem(sys.modules, "qlib", qlib_stub)
    monkeypatch.setitem(sys.modules, "qlib.data", qlib_data_stub)

    script = Path(__file__).resolve().parents[1] / "scripts" / "check_data_health.py"
    spec = importlib.util.spec_from_file_location("testable_data_health_checker", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DataHealthChecker


def write_csv(directory, data, filename="sample.csv"):
    pd.DataFrame(data).to_csv(directory / filename, index=False)


def test_csv_mode_does_not_require_qlib_dir(checker_class, tmp_path):
    write_csv(tmp_path, {"open": [10], "high": [11], "low": [9], "close": [10], "volume": [100], "factor": [1]})

    checker = checker_class(csv_path=str(tmp_path))

    assert checker.qlib_dir is None
    assert list(checker.data) == ["sample.csv"]
    assert checker.check_features_dir_lowercase() is None


def test_missing_values_alone_trigger_summary(checker_class, tmp_path, capsys):
    write_csv(
        tmp_path,
        {
            "open": [10, 10],
            "high": [11, 11],
            "low": [9, 9],
            "close": [10, 10],
            "volume": [100, None],
            "factor": [1, 1],
        },
    )

    checker_class(csv_path=str(tmp_path)).check_data()

    output = capsys.readouterr().out
    assert "Summary of data health check" in output
    assert "sample.csv" in output


def test_missing_factor_and_multiple_required_columns_are_reported(checker_class, tmp_path, capsys):
    write_csv(tmp_path, {"close": [10, 10], "volume": [100, 100]})
    checker = checker_class(csv_path=str(tmp_path))

    missing_columns = checker.check_required_columns()
    assert set(missing_columns["missing_col"]) == {"open", "high", "low"}

    missing_factor = checker.check_missing_factor()
    assert missing_factor.loc["sample.csv", "missing_factor_col"]
    assert not missing_factor.loc["sample.csv", "missing_factor_data"]

    checker.check_data()
    assert "Summary of data health check" in capsys.readouterr().out


def test_empty_factor_column_is_reported(checker_class, tmp_path):
    write_csv(tmp_path, {"open": [10], "high": [11], "low": [9], "close": [10], "volume": [100], "factor": [None]})

    missing_factor = checker_class(csv_path=str(tmp_path)).check_missing_factor()

    assert not missing_factor.loc["sample.csv", "missing_factor_col"]
    assert missing_factor.loc["sample.csv", "missing_factor_data"]


def test_missing_values_do_not_require_every_ohlcv_column(checker_class, tmp_path):
    write_csv(tmp_path, {"close": [10, None], "volume": [100, 100], "factor": [1, 1]})

    missing = checker_class(csv_path=str(tmp_path)).check_missing_data()

    assert missing.loc["sample.csv", "close"] == 1
    assert missing.loc["sample.csv", "open"] == 0


def test_large_price_change_uses_csv_date_column(checker_class, tmp_path):
    write_csv(
        tmp_path,
        {
            "date": ["2026-01-01", "2026-01-02"],
            "open": [10, 20],
            "high": [11, 22],
            "low": [9, 18],
            "close": [10, 20],
            "volume": [100, 100],
            "factor": [1, 1],
        },
    )

    large_steps = checker_class(csv_path=str(tmp_path)).check_large_step_changes()

    assert set(large_steps["date"]) == {"2026-01-02"}


def test_large_price_change_preserves_qlib_multiindex_date(checker_class, tmp_path):
    write_csv(tmp_path, {"close": [10]})
    checker = checker_class(csv_path=str(tmp_path))
    index = pd.MultiIndex.from_product([["SH600000"], pd.to_datetime(["2026-01-01", "2026-01-02"])])
    checker.data = {"SH600000": pd.DataFrame({"close": [10, 20]}, index=index)}

    large_steps = checker.check_large_step_changes()

    assert large_steps.loc["SH600000", "date"] == "2026-01-02"


def test_missing_extra_column_is_not_reported(checker_class, tmp_path):
    write_csv(
        tmp_path,
        {
            "open": [10, 10],
            "high": [11, 11],
            "low": [9, 9],
            "close": [10, 10],
            "volume": [100, 100],
            "factor": [1, 1],
            "extra": [None, None],
        },
    )

    checker = checker_class(csv_path=str(tmp_path))
    assert checker.check_missing_data() is None


def test_empty_factor_does_not_trigger_ohlcv_missing_data(
    checker_class, tmp_path, capsys
):
    write_csv(
        tmp_path,
        {
            "open": [10, 10],
            "high": [11, 11],
            "low": [9, 9],
            "close": [10, 10],
            "volume": [100, 100],
            "factor": [None, None],
        },
    )

    checker = checker_class(csv_path=str(tmp_path))

    assert checker.check_missing_data() is None

    factor_result = checker.check_missing_factor()
    assert factor_result.loc["sample.csv", "missing_factor_data"]

    checker.check_data()
    output = capsys.readouterr().out

    assert "Summary of data health check" in output
    assert "missing_factor_data" in output

def test_summary_reports_missing_columns_before_nan_counts(
    checker_class, tmp_path, capsys
):
    write_csv(
        tmp_path,
        {
            "open": [10, None],
            "low": [9, 9],
            "close": [10, 10],
            "volume": [100, 100],
            "factor": [1, 1],
        },
    )

    checker = checker_class(csv_path=str(tmp_path))
    required_columns = checker.check_required_columns()
    missing_data = checker.check_missing_data()

    assert required_columns.loc["sample.csv", "missing_col"] == "high"
    assert missing_data.loc["sample.csv", "open"] == 1

    checker.check_data()
    output = capsys.readouterr().out

    required_table = str(required_columns)
    missing_data_table = str(missing_data)

    assert required_table in output
    assert missing_data_table in output
    assert output.index(required_table) < output.index(missing_data_table)
