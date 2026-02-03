# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for the qlib telemetry module."""

import time
import threading

import pytest

from qlib.utils.telemetry import (
    QlibMetrics,
    QlibTracer,
    InMemoryBackend,
    LoggingBackend,
    MetricEvent,
    SpanEvent,
    metrics,
    tracer,
    enable_inmemory_backend,
    enable_logging_backend,
)


class TestMetricEvent:
    def test_default_timestamp(self):
        event = MetricEvent(name="test", value=1.0, metric_type="counter")
        assert event.timestamp > 0
        assert event.tags == {}

    def test_with_tags(self):
        event = MetricEvent(name="test", value=42.0, metric_type="gauge", tags={"env": "prod"})
        assert event.tags == {"env": "prod"}


class TestSpanEvent:
    def test_default_values(self):
        span = SpanEvent(name="test_span", duration_s=1.5)
        assert span.parent is None
        assert span.error is None
        assert span.tags == {}


class TestQlibMetrics:
    def setup_method(self):
        self.m = QlibMetrics()

    def test_no_backend_is_noop(self):
        # Should not raise
        self.m.counter("test")
        self.m.gauge("test", 1.0)
        self.m.histogram("test", 1.0)
        assert not self.m.enabled

    def test_add_backend(self):
        backend = InMemoryBackend()
        self.m.add_backend(backend)
        assert self.m.enabled

    def test_counter(self):
        backend = InMemoryBackend()
        self.m.add_backend(backend)
        self.m.counter("requests", 1, tags={"method": "GET"})
        assert len(backend.metrics) == 1
        assert backend.metrics[0].name == "requests"
        assert backend.metrics[0].value == 1
        assert backend.metrics[0].metric_type == "counter"
        assert backend.metrics[0].tags == {"method": "GET"}

    def test_gauge(self):
        backend = InMemoryBackend()
        self.m.add_backend(backend)
        self.m.gauge("memory_mb", 512.5)
        assert backend.metrics[0].metric_type == "gauge"
        assert backend.metrics[0].value == 512.5

    def test_histogram(self):
        backend = InMemoryBackend()
        self.m.add_backend(backend)
        self.m.histogram("latency_ms", 42.3)
        assert backend.metrics[0].metric_type == "histogram"

    def test_multiple_backends(self):
        b1 = InMemoryBackend()
        b2 = InMemoryBackend()
        self.m.add_backend(b1)
        self.m.add_backend(b2)
        self.m.counter("test")
        assert len(b1.metrics) == 1
        assert len(b2.metrics) == 1

    def test_remove_backend(self):
        backend = InMemoryBackend()
        self.m.add_backend(backend)
        self.m.remove_backend(backend)
        assert not self.m.enabled
        self.m.counter("test")
        assert len(backend.metrics) == 0

    def test_clear_backends(self):
        self.m.add_backend(InMemoryBackend())
        self.m.add_backend(InMemoryBackend())
        self.m.clear_backends()
        assert not self.m.enabled

    def test_backend_error_does_not_propagate(self):
        class FailingBackend(InMemoryBackend):
            def record_metric(self, event):
                raise RuntimeError("backend error")

        self.m.add_backend(FailingBackend())
        # Should not raise
        self.m.counter("test")


class TestQlibTracer:
    def setup_method(self):
        self.m = QlibMetrics()
        self.t = QlibTracer(self.m)

    def test_span_noop_without_backend(self):
        # Should not raise, should not record
        with self.t.span("test"):
            pass

    def test_span_records_duration(self):
        backend = InMemoryBackend()
        self.m.add_backend(backend)
        with self.t.span("slow_op"):
            time.sleep(0.05)
        assert len(backend.spans) == 1
        assert backend.spans[0].name == "slow_op"
        assert backend.spans[0].duration_s >= 0.04
        assert backend.spans[0].error is None

    def test_span_with_tags(self):
        backend = InMemoryBackend()
        self.m.add_backend(backend)
        with self.t.span("tagged_op", tags={"freq": "day"}):
            pass
        assert backend.spans[0].tags == {"freq": "day"}

    def test_span_records_error(self):
        backend = InMemoryBackend()
        self.m.add_backend(backend)
        with pytest.raises(ValueError):
            with self.t.span("failing_op"):
                raise ValueError("test error")
        assert len(backend.spans) == 1
        assert backend.spans[0].error == "test error"

    def test_nested_spans_have_parent(self):
        backend = InMemoryBackend()
        self.m.add_backend(backend)
        with self.t.span("parent"):
            with self.t.span("child"):
                pass
        # child should have parent
        child_span = next(s for s in backend.spans if s.name == "child")
        parent_span = next(s for s in backend.spans if s.name == "parent")
        assert child_span.parent == "parent"
        assert parent_span.parent is None

    def test_span_emits_histogram_metric(self):
        backend = InMemoryBackend()
        self.m.add_backend(backend)
        with self.t.span("timed_op"):
            pass
        histograms = [m for m in backend.metrics if m.metric_type == "histogram"]
        assert len(histograms) == 1
        assert histograms[0].tags["span"] == "timed_op"

    def test_traced_decorator(self):
        backend = InMemoryBackend()
        self.m.add_backend(backend)

        @self.t.traced("my_func")
        def my_func(x):
            return x * 2

        result = my_func(21)
        assert result == 42
        assert len(backend.spans) == 1
        assert backend.spans[0].name == "my_func"

    def test_thread_safety(self):
        backend = InMemoryBackend()
        self.m.add_backend(backend)
        errors = []

        def worker(name):
            try:
                with self.t.span(f"thread_{name}"):
                    time.sleep(0.01)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(backend.spans) == 10


class TestInMemoryBackend:
    def test_get_metrics_filtered(self):
        backend = InMemoryBackend()
        backend.record_metric(MetricEvent(name="a", value=1, metric_type="counter"))
        backend.record_metric(MetricEvent(name="b", value=2, metric_type="counter"))
        backend.record_metric(MetricEvent(name="a", value=3, metric_type="counter"))
        assert len(backend.get_metrics("a")) == 2
        assert len(backend.get_metrics("b")) == 1
        assert len(backend.get_metrics()) == 3

    def test_get_spans_filtered(self):
        backend = InMemoryBackend()
        backend.record_span(SpanEvent(name="x", duration_s=1.0))
        backend.record_span(SpanEvent(name="y", duration_s=2.0))
        assert len(backend.get_spans("x")) == 1
        assert len(backend.get_spans()) == 2

    def test_clear(self):
        backend = InMemoryBackend()
        backend.record_metric(MetricEvent(name="a", value=1, metric_type="counter"))
        backend.record_span(SpanEvent(name="x", duration_s=1.0))
        backend.clear()
        assert len(backend.metrics) == 0
        assert len(backend.spans) == 0

    def test_summary(self):
        backend = InMemoryBackend()
        backend.record_span(SpanEvent(name="op", duration_s=1.0))
        backend.record_span(SpanEvent(name="op", duration_s=3.0))
        backend.record_metric(MetricEvent(name="hits", value=1, metric_type="counter"))

        summary = backend.summary()
        assert summary["spans"]["op"]["count"] == 2
        assert summary["spans"]["op"]["avg_s"] == 2.0
        assert summary["spans"]["op"]["min_s"] == 1.0
        assert summary["spans"]["op"]["max_s"] == 3.0
        assert summary["metrics"]["hits"]["count"] == 1


class TestLoggingBackend:
    def test_record_metric_does_not_raise(self):
        backend = LoggingBackend()
        backend.record_metric(MetricEvent(name="test", value=1, metric_type="counter"))

    def test_record_span_does_not_raise(self):
        backend = LoggingBackend()
        backend.record_span(SpanEvent(name="test", duration_s=1.0))


class TestModuleLevelSingletons:
    def setup_method(self):
        metrics.clear_backends()

    def teardown_method(self):
        metrics.clear_backends()

    def test_metrics_singleton_works(self):
        backend = enable_inmemory_backend()
        metrics.counter("singleton_test")
        assert len(backend.metrics) == 1

    def test_tracer_singleton_works(self):
        backend = enable_inmemory_backend()
        with tracer.span("singleton_span"):
            pass
        assert len(backend.spans) == 1

    def test_enable_logging_backend(self):
        backend = enable_logging_backend()
        metrics.counter("log_test")
        # Just verify it doesn't crash
        metrics.remove_backend(backend)
