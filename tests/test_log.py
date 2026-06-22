import io
import json
import logging
from contextlib import contextmanager

from qlib.log import JSONFormatter, get_module_logger


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
