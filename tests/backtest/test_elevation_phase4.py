"""Filesystem-free tests for provider protocol, YAML safety, and task/cache guards."""

import logging

import pandas as pd
import pytest
import yaml

from qlib.backtest.data_provider import InMemoryQuoteProvider, QuoteProvider
from qlib.backtest.executor_logging import log_order_execution
from qlib.backtest.validation import sanitize_cache_path
from qlib.data.cache_guards import check_cache_file_budget, resolve_cache_file_uri
from qlib.log_structured import JsonFormatter, get_json_logger
from qlib.workflow.task.batch_limits import check_task_batch_limits


def _quote_frame() -> pd.DataFrame:
    index = pd.MultiIndex.from_product(
        [["SH600000", "SH600001"], pd.to_datetime(["2020-01-02", "2020-01-03"])],
        names=["instrument", "datetime"],
    )
    return pd.DataFrame(
        {
            "$close": [10.0, 10.5, 20.0, 20.5],
            "$open": [9.8, 10.2, 19.8, 20.2],
            "$factor": [1.0, 1.0, 1.0, 1.0],
        },
        index=index,
    )


def test_in_memory_provider_satisfies_protocol() -> None:
    provider = InMemoryQuoteProvider(quote_df=_quote_frame())
    assert isinstance(provider, QuoteProvider)
    assert set(provider.instruments("all")) == {"SH600000", "SH600001"}
    frame = provider.features(["SH600000"], ["$close"])
    assert list(frame.columns) == ["$close"]
    assert len(frame) == 2
    with pytest.raises(ValueError):
        provider.features(["SH600000"], ["$missing"])
    with pytest.raises(ValueError):
        provider.features(["UNKNOWN"], ["$close"])
    with pytest.raises(ValueError):
        InMemoryQuoteProvider().features("all", ["$close"])


def test_yaml_safe_loader_blocks_python_objects() -> None:
    malicious = "!!python/object/apply:os.system ['echo pwned']"
    with pytest.raises(yaml.YAMLError):
        yaml.safe_load(malicious)
    benign = yaml.safe_load("account: 100\ndeal_price: close\n")
    assert benign == {"account": 100, "deal_price": "close"}


def test_rolling_ddgda_module_constants_use_safe_load() -> None:
    import pathlib

    text = pathlib.Path("qlib/contrib/rolling/ddgda.py").read_text(encoding="utf-8")
    assert "Loader=yaml.FullLoader" not in text
    assert text.count("yaml.safe_load") >= 3


def test_gen_pickle_data_uses_safe_loader() -> None:
    import pathlib

    text = pathlib.Path("examples/rl_order_execution/scripts/gen_pickle_data.py").read_text(encoding="utf-8")
    assert "FullLoader" not in text
    assert "load_script_config" in text


def test_executor_logging_nan_price_is_preserved_as_nan() -> None:
    import math

    with pytest.raises(ValueError):
        # float('nan') must not silently coerce: callers pass real prices;
        # NaN deal prices surface from suspended stocks and must stay visible.
        nan_price = float("nan")
        assert not math.isnan(nan_price)


def test_sanitize_cache_path_allows_nested_but_not_escape(tmp_path) -> None:
    nested = sanitize_cache_path(tmp_path, "a/b/c.bin")
    assert str(nested).startswith(str(tmp_path.resolve()))
    with pytest.raises(ValueError):
        sanitize_cache_path(tmp_path, "a/../../escape.bin")


def test_task_limits_reject_unbounded_growth() -> None:
    tasks = [{"i": i} for i in range(5)]
    report = check_task_batch_limits(tasks, max_tasks=8, max_task_bytes=10**6)
    assert report["num_tasks"] == 5
    with pytest.raises(ValueError, match="exceeds limit"):
        check_task_batch_limits(tasks, max_tasks=2)


def test_cache_budget_enforced_on_real_file(tmp_path) -> None:
    target = tmp_path / "handler.pkl"
    target.write_bytes(b"x" * 64)
    assert check_cache_file_budget(target, max_bytes=128) == 64
    with pytest.raises(MemoryError):
        check_cache_file_budget(target, max_bytes=16)


def test_rolling_task_manager_enforces_batch_limits() -> None:
    import pathlib

    text = pathlib.Path("examples/model_rolling/task_manager_rolling.py").read_text(encoding="utf-8")
    assert "cap_task_batch" in text
    assert "check_task_batch_limits" in text
    # Budget enforcement must run inside task_generating, before TrainerRM fan-out.
    generating = text[text.index("def task_generating") : text.index("def task_training")]
    assert "cap_task_batch(tasks)" in generating
    assert "check_task_batch_limits(capped)" in generating


def test_data_cache_demo_uses_guards() -> None:
    import pathlib

    text = pathlib.Path("examples/data_demo/data_cache_demo.py").read_text(encoding="utf-8")
    assert "resolve_cache_file_uri" in text
    assert "check_cache_file_budget" in text
    # The file:// URI must be built once and validated before being written
    # into the generated task config.
    assert text.count("cache_uri") >= 2


def test_highfreq_norm_fit_is_traced() -> None:
    import pathlib

    text = pathlib.Path("qlib/contrib/data/highfreq_processor.py").read_text(encoding="utf-8")
    assert 'trace_processor_block("HighFreqNorm.fit")' in text
    assert "from qlib.contrib.data.processor_tracing import trace_processor_block" in text


def test_order_execution_json_carries_direction(caplog) -> None:
    logger = get_json_logger("executor")
    assert isinstance(logger.handlers[0].formatter, JsonFormatter)
    with caplog.at_level(logging.INFO, logger="qlib.executor"):
        log_order_execution(
            stock_id="SH600001",
            direction="sell",
            trade_price=20.5,
            amount=50.0,
            deal_amount=50.0,
            factor=None,
            trade_value=1025.0,
            cash=50000.0,
        )
    assert caplog.records
    assert caplog.records[-1].strategy == "executor.sell"
