import io
import json
import logging
from contextlib import contextmanager

from qlib.log import JSONFormatter, get_module_logger, set_log_with_config


@contextmanager
def _logger_with_handler(name, formatter):
    logger = get_module_logger(name, level=logging.INFO)
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(formatter)
    logger.logger.handlers = [handler]
    logger.logger.propagate = False
    try:
        yield logger, stream
    finally:
        logger.logger.handlers = []


def test_default_logging_remains_plain_text():
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    with _logger_with_handler("test.default_logging", formatter) as (logger, stream):
        logger.info("Dataset loaded")

        assert stream.getvalue().strip() == "INFO: Dataset loaded"


def test_json_formatter_produces_structured_output():
    with _logger_with_handler("test.structured_logging", JSONFormatter()) as (logger, stream):
        logger.info("Dataset loaded")

        output = json.loads(stream.getvalue())
        assert output["level"] == "INFO"
        assert output["logger"] == "qlib.test.structured_logging"
        assert output["message"] == "Dataset loaded"
        assert "timestamp" in output


def test_json_formatter_includes_extra_fields():
    with _logger_with_handler("test.structured_extra", JSONFormatter()) as (logger, stream):
        logger.info("Dataset loaded", extra={"dataset_size": 100, "cache_hit": True})

        output = json.loads(stream.getvalue())
        assert output["dataset_size"] == 100
        assert output["cache_hit"] is True


def test_json_formatter_includes_exception_details():
    with _logger_with_handler("test.structured_exception", JSONFormatter()) as (logger, stream):
        try:
            raise ValueError("bad dataset")
        except ValueError:
            logger.exception("Dataset load failed")

        output = json.loads(stream.getvalue())
        assert output["level"] == "ERROR"
        assert output["message"] == "Dataset load failed"
        assert "ValueError: bad dataset" in output["exception"]


def test_structured_logging_can_be_enabled_with_compact_config(capsys):
    qlib_logger = logging.getLogger("qlib")
    original_handlers = qlib_logger.handlers[:]
    original_level = qlib_logger.level
    original_propagate = qlib_logger.propagate

    try:
        set_log_with_config({"structured": True, "format": "json"})
        logger = get_module_logger("test.configured_structured", level=logging.INFO)
        logger.info("Configured JSON logging", extra={"run_id": "abc123"})

        captured = capsys.readouterr()
        output = captured.err.strip() or captured.out.strip()
        log_data = json.loads(output.splitlines()[-1])

        assert log_data["level"] == "INFO"
        assert log_data["logger"] == "qlib.test.configured_structured"
        assert log_data["message"] == "Configured JSON logging"
        assert log_data["run_id"] == "abc123"
    finally:
        qlib_logger.handlers = original_handlers
        qlib_logger.setLevel(original_level)
        qlib_logger.propagate = original_propagate
