# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional


@dataclass(frozen=True)
class TraceSpan:
    """A lightweight trace span used to connect related Qlib logs."""

    name: str
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None

    def to_log_fields(self) -> Dict[str, str]:
        fields = {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "span_name": self.name,
        }
        if self.parent_span_id is not None:
            fields["parent_span_id"] = self.parent_span_id
        return fields


_TRACING_ENABLED = False
_CURRENT_SPAN: ContextVar[Optional[TraceSpan]] = ContextVar("qlib_current_trace_span", default=None)


def configure_tracing(tracing_config: Optional[Dict[str, Any]] = None) -> None:
    """Configure lightweight Qlib tracing.

    Tracing is disabled by default. Passing ``{"enabled": True}`` enables span
    creation through :func:`trace_span`.
    """
    global _TRACING_ENABLED

    tracing_config = tracing_config or {}
    _TRACING_ENABLED = bool(tracing_config.get("enabled", False))
    if not _TRACING_ENABLED:
        _CURRENT_SPAN.set(None)


def is_tracing_enabled() -> bool:
    return _TRACING_ENABLED


def get_current_span() -> Optional[TraceSpan]:
    if not _TRACING_ENABLED:
        return None
    return _CURRENT_SPAN.get()


def get_current_trace_context() -> Dict[str, str]:
    span = get_current_span()
    return {} if span is None else span.to_log_fields()


@contextmanager
def trace_span(name: str, trace_id: Optional[str] = None) -> Iterator[Optional[TraceSpan]]:
    """Create a lightweight trace span when tracing is enabled.

    When tracing is disabled, this context manager yields ``None`` and has no
    logging side effects.
    """
    if not _TRACING_ENABLED:
        yield None
        return

    parent_span = _CURRENT_SPAN.get()
    span = TraceSpan(
        name=name,
        trace_id=trace_id or (parent_span.trace_id if parent_span is not None else uuid.uuid4().hex),
        span_id=uuid.uuid4().hex,
        parent_span_id=parent_span.span_id if parent_span is not None else None,
    )
    token = _CURRENT_SPAN.set(span)
    try:
        yield span
    finally:
        _CURRENT_SPAN.reset(token)
