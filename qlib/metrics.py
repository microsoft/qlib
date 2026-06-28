# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from threading import RLock
from typing import Any, Dict, Iterator, Optional


class NoOpMetricsRecorder:
    """Metrics recorder used when observability metrics are disabled."""

    enabled = False

    def increment(self, name: str, value: float = 1, tags: Optional[Dict[str, Any]] = None) -> None:
        return None

    def gauge(self, name: str, value: float, tags: Optional[Dict[str, Any]] = None) -> None:
        return None

    def timing(self, name: str, value: float, tags: Optional[Dict[str, Any]] = None) -> None:
        return None

    @contextmanager
    def timer(self, name: str, tags: Optional[Dict[str, Any]] = None) -> Iterator[None]:
        yield None

    def snapshot(self) -> Dict[str, Dict[str, Any]]:
        return {"counters": {}, "gauges": {}, "timings": {}}

    def reset(self) -> None:
        return None


class InMemoryMetricsRecorder:
    """Small in-memory metrics recorder for lightweight Qlib observability.

    This recorder intentionally has no external service dependency. It gives
    Qlib a testable metrics foundation that can later be exported to logs,
    experiment recorders, or monitoring backends.
    """

    enabled = True

    def __init__(self) -> None:
        self._lock = RLock()
        self._counters = defaultdict(float)
        self._gauges = {}
        self._timings = {}

    @staticmethod
    def _metric_key(name: str, tags: Optional[Dict[str, Any]] = None) -> str:
        if not tags:
            return name
        tag_str = ",".join(f"{key}={tags[key]}" for key in sorted(tags))
        return f"{name}{{{tag_str}}}"

    def increment(self, name: str, value: float = 1, tags: Optional[Dict[str, Any]] = None) -> None:
        key = self._metric_key(name, tags)
        with self._lock:
            self._counters[key] += value

    def gauge(self, name: str, value: float, tags: Optional[Dict[str, Any]] = None) -> None:
        key = self._metric_key(name, tags)
        with self._lock:
            self._gauges[key] = value

    def timing(self, name: str, value: float, tags: Optional[Dict[str, Any]] = None) -> None:
        key = self._metric_key(name, tags)
        with self._lock:
            stats = self._timings.setdefault(
                key,
                {
                    "count": 0,
                    "total": 0.0,
                    "last": None,
                    "min": None,
                    "max": None,
                },
            )
            stats["count"] += 1
            stats["total"] += value
            stats["last"] = value
            stats["min"] = value if stats["min"] is None else min(stats["min"], value)
            stats["max"] = value if stats["max"] is None else max(stats["max"], value)

    @contextmanager
    def timer(self, name: str, tags: Optional[Dict[str, Any]] = None) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield None
        finally:
            self.timing(name, time.perf_counter() - start, tags=tags)

    def snapshot(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "timings": {name: dict(stats) for name, stats in self._timings.items()},
            }

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._timings.clear()


_NOOP_RECORDER = NoOpMetricsRecorder()
_METRICS_RECORDER = _NOOP_RECORDER


def configure_metrics(metrics_config: Optional[Dict[str, Any]] = None) -> None:
    """Configure Qlib's process-local metrics recorder.

    Metrics are disabled by default. Passing ``{"enabled": True}`` enables an
    in-memory recorder that can be inspected through ``snapshot()``.
    """
    global _METRICS_RECORDER

    metrics_config = metrics_config or {}
    if metrics_config.get("enabled", False):
        _METRICS_RECORDER = InMemoryMetricsRecorder()
    else:
        _METRICS_RECORDER = _NOOP_RECORDER


def get_metrics_recorder():
    """Return the active metrics recorder."""
    return _METRICS_RECORDER
