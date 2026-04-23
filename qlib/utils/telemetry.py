# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Qlib Telemetry Module
=====================

Provides structured metrics collection, workflow tracing, and performance
monitoring for Qlib's data pipeline, model training, and backtest workflows.

This module introduces three core abstractions:

- **QlibMetrics**: A singleton metrics collector that tracks counters, gauges,
  histograms, and timers. Backends can be registered to export metrics
  (e.g., to console, file, or external systems).

- **QlibTracer**: A context-manager-based tracer for tracking workflow spans
  with parent-child relationships. Compatible with ``TimeInspector`` but
  adds span hierarchy, metadata, and backend-agnostic export.

- **Backends**: Pluggable metric/trace sinks. Ships with ``LoggingBackend``
  (uses existing ``get_module_logger``) and ``InMemoryBackend`` (for testing
  and programmatic access).

Design Principles:

- **Zero overhead by default**: Telemetry is opt-in. When no backend is
  registered, all operations are no-ops with negligible cost.
- **Non-invasive**: Instrumentation uses decorators and context managers.
  No changes to function signatures or return values.
- **Backward compatible**: Works alongside existing ``TimeInspector`` and
  ``get_module_logger`` without conflicts.
- **Extensible**: Users can register custom backends (e.g., Prometheus,
  OpenTelemetry, Datadog) by implementing the ``MetricsBackend`` protocol.

Example usage::

    from qlib.utils.telemetry import metrics, tracer

    # Record a counter
    metrics.counter("cache.hits", 1, tags={"cache": "expression"})

    # Record a gauge
    metrics.gauge("memory.rss_mb", 1024.5)

    # Time a code block
    with tracer.span("data_loading", tags={"freq": "day"}):
        data = load_data()

    # Use as a decorator
    @tracer.traced("model_training")
    def train_model():
        ...
"""

import time
import threading
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from qlib.log import get_module_logger

logger = get_module_logger("telemetry")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MetricEvent:
    """A single metric measurement."""

    name: str
    value: float
    metric_type: str  # "counter", "gauge", "histogram"
    tags: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class SpanEvent:
    """A single trace span representing a unit of work."""

    name: str
    duration_s: float
    tags: Dict[str, str] = field(default_factory=dict)
    parent: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------

class MetricsBackend(ABC):
    """Interface for metrics/trace export backends."""

    @abstractmethod
    def record_metric(self, event: MetricEvent) -> None:
        """Record a metric event."""

    @abstractmethod
    def record_span(self, span: SpanEvent) -> None:
        """Record a completed span."""

    def flush(self) -> None:
        """Flush any buffered data. Optional."""


class LoggingBackend(MetricsBackend):
    """Backend that logs metrics and spans via ``get_module_logger``.

    This integrates with Qlib's existing logging infrastructure, so
    metrics appear in the same log stream as other Qlib messages.
    """

    def __init__(self, level: int = logging.DEBUG):
        self._level = level

    def record_metric(self, event: MetricEvent) -> None:
        tags_str = " ".join(f"{k}={v}" for k, v in event.tags.items())
        logger.log(
            self._level,
            "[metric] %s=%s type=%s %s",
            event.name,
            event.value,
            event.metric_type,
            tags_str,
        )

    def record_span(self, span: SpanEvent) -> None:
        tags_str = " ".join(f"{k}={v}" for k, v in span.tags.items())
        status = "ERROR" if span.error else "OK"
        logger.log(
            self._level,
            "[span] %s duration=%.3fs status=%s %s",
            span.name,
            span.duration_s,
            status,
            tags_str,
        )


class InMemoryBackend(MetricsBackend):
    """Backend that stores metrics and spans in memory for programmatic access.

    Useful for testing, assertions, and building dashboards.
    """

    def __init__(self):
        self.metrics: List[MetricEvent] = []
        self.spans: List[SpanEvent] = []
        self._lock = threading.Lock()

    def record_metric(self, event: MetricEvent) -> None:
        with self._lock:
            self.metrics.append(event)

    def record_span(self, span: SpanEvent) -> None:
        with self._lock:
            self.spans.append(span)

    def get_metrics(self, name: Optional[str] = None) -> List[MetricEvent]:
        """Get recorded metrics, optionally filtered by name."""
        with self._lock:
            if name is None:
                return list(self.metrics)
            return [m for m in self.metrics if m.name == name]

    def get_spans(self, name: Optional[str] = None) -> List[SpanEvent]:
        """Get recorded spans, optionally filtered by name."""
        with self._lock:
            if name is None:
                return list(self.spans)
            return [s for s in self.spans if s.name == name]

    def clear(self) -> None:
        """Clear all recorded data."""
        with self._lock:
            self.metrics.clear()
            self.spans.clear()

    def summary(self) -> Dict[str, Any]:
        """Return a summary of collected metrics and spans."""
        with self._lock:
            metric_counts = defaultdict(int)
            for m in self.metrics:
                metric_counts[m.name] += 1

            span_stats = defaultdict(list)
            for s in self.spans:
                span_stats[s.name].append(s.duration_s)

            return {
                "metrics": {
                    name: {"count": count} for name, count in metric_counts.items()
                },
                "spans": {
                    name: {
                        "count": len(durations),
                        "total_s": sum(durations),
                        "avg_s": sum(durations) / len(durations) if durations else 0,
                        "min_s": min(durations) if durations else 0,
                        "max_s": max(durations) if durations else 0,
                    }
                    for name, durations in span_stats.items()
                },
            }


# ---------------------------------------------------------------------------
# QlibMetrics - Singleton metrics collector
# ---------------------------------------------------------------------------

class QlibMetrics:
    """Centralized metrics collector with pluggable backends.

    All methods are safe to call even when no backend is registered;
    they simply become no-ops.
    """

    def __init__(self):
        self._backends: List[MetricsBackend] = []
        self._lock = threading.Lock()

    def add_backend(self, backend: MetricsBackend) -> None:
        """Register a metrics backend."""
        with self._lock:
            self._backends.append(backend)

    def remove_backend(self, backend: MetricsBackend) -> None:
        """Remove a metrics backend."""
        with self._lock:
            self._backends.remove(backend)

    def clear_backends(self) -> None:
        """Remove all backends."""
        with self._lock:
            self._backends.clear()

    @property
    def enabled(self) -> bool:
        """Whether any backend is registered."""
        return len(self._backends) > 0

    def _emit(self, event: MetricEvent) -> None:
        if not self._backends:
            return
        for backend in self._backends:
            try:
                backend.record_metric(event)
            except Exception:
                pass  # Never let telemetry crash the application

    def counter(self, name: str, value: float = 1, tags: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter metric."""
        self._emit(MetricEvent(name=name, value=value, metric_type="counter", tags=tags or {}))

    def gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Set a gauge metric to a specific value."""
        self._emit(MetricEvent(name=name, value=value, metric_type="gauge", tags=tags or {}))

    def histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a histogram value (e.g., latency, size)."""
        self._emit(MetricEvent(name=name, value=value, metric_type="histogram", tags=tags or {}))

    def flush(self) -> None:
        """Flush all backends."""
        for backend in self._backends:
            try:
                backend.flush()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# QlibTracer - Workflow span tracer
# ---------------------------------------------------------------------------

class QlibTracer:
    """Context-manager-based tracer for tracking workflow spans.

    Supports parent-child relationships via thread-local storage.
    """

    def __init__(self, metrics_collector: QlibMetrics):
        self._metrics = metrics_collector
        self._local = threading.local()

    def _get_parent(self) -> Optional[str]:
        stack = getattr(self._local, "span_stack", [])
        return stack[-1] if stack else None

    def _push_span(self, name: str) -> None:
        if not hasattr(self._local, "span_stack"):
            self._local.span_stack = []
        self._local.span_stack.append(name)

    def _pop_span(self) -> None:
        stack = getattr(self._local, "span_stack", [])
        if stack:
            stack.pop()

    @contextmanager
    def span(self, name: str, tags: Optional[Dict[str, str]] = None):
        """Context manager that traces a span of work.

        Example::

            with tracer.span("load_data", tags={"freq": "day"}):
                data = load_data()
        """
        if not self._metrics.enabled:
            yield
            return

        parent = self._get_parent()
        self._push_span(name)
        start = time.perf_counter()
        error = None
        try:
            yield
        except Exception as e:
            error = str(e)
            raise
        finally:
            duration = time.perf_counter() - start
            self._pop_span()

            span_event = SpanEvent(
                name=name,
                duration_s=duration,
                tags=tags or {},
                parent=parent,
                error=error,
            )
            for backend in self._metrics._backends:
                try:
                    backend.record_span(span_event)
                except Exception:
                    pass

            # Also emit as a histogram metric for aggregation
            self._metrics.histogram(
                f"span.duration_s",
                duration,
                tags={"span": name, **(tags or {})},
            )

    def traced(self, name: str, tags: Optional[Dict[str, str]] = None) -> Callable:
        """Decorator that traces a function call as a span.

        Example::

            @tracer.traced("model_fit")
            def fit(self, dataset):
                ...
        """

        def decorator(func):
            def wrapper(*args, **kwargs):
                with self.span(name, tags=tags):
                    return func(*args, **kwargs)
            wrapper.__name__ = func.__name__
            wrapper.__doc__ = func.__doc__
            return wrapper

        return decorator


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

#: Global metrics collector. Import and use directly:
#: ``from qlib.utils.telemetry import metrics``
metrics = QlibMetrics()

#: Global tracer. Import and use directly:
#: ``from qlib.utils.telemetry import tracer``
tracer = QlibTracer(metrics)


def enable_logging_backend(level: int = logging.DEBUG) -> LoggingBackend:
    """Convenience function to enable the logging backend.

    Returns the backend instance for later removal if needed.
    """
    backend = LoggingBackend(level=level)
    metrics.add_backend(backend)
    return backend


def enable_inmemory_backend() -> InMemoryBackend:
    """Convenience function to enable the in-memory backend.

    Returns the backend instance for direct access to collected data.
    """
    backend = InMemoryBackend()
    metrics.add_backend(backend)
    return backend
