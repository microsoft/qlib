# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Structured tracing + memory guards for high-frequency data processing.

Additive helpers only: existing ``HighFreqNorm``/``HighFreqTrans`` behavior is
unchanged. Wrap hot blocks with :func:`trace_processor_block` to get JSON log
lines carrying latency, and call :func:`estimate_frame_bytes` /
:func:`check_frame_memory_limit` before materializing large CSI300/500 frames.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

import numpy as np
import pandas as pd

from qlib.log_structured import get_json_logger

_processor_logger = get_json_logger("contrib.data.processor")


@contextmanager
def trace_processor_block(
    processor: str,
    *,
    run_id: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Iterator[None]:
    """Trace one high-frequency processor block with structured latency logging."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        fields: Dict[str, Any] = {"processor": processor, "latency_ms": round(elapsed_ms, 3)}
        if run_id is not None:
            fields["run_id"] = run_id
        if extra:
            fields.update(extra)
        _processor_logger.info("hf_processor.block", extra={"extra": fields})


def estimate_frame_bytes(frame: pd.DataFrame) -> int:
    """Estimate in-memory bytes of a feature frame (index + values)."""
    values_bytes = int(np.asarray(frame.values).nbytes) if len(frame) else 0
    try:
        index_bytes = int(frame.index.memory_usage(deep=True))
    except Exception:  # pragma: no cover - index impl detail
        index_bytes = 0
    return values_bytes + index_bytes


def check_frame_memory_limit(frame: pd.DataFrame, *, max_bytes: int) -> Dict[str, Any]:
    """Validate a frame against an explicit memory budget.

    Raises
    ------
    MemoryError
        If the estimated footprint exceeds ``max_bytes``.
    """
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    used = estimate_frame_bytes(frame)
    report = {"estimated_bytes": used, "max_bytes": max_bytes, "rows": len(frame)}
    if used > max_bytes:
        raise MemoryError(
            f"feature frame uses ~{used} bytes, exceeding budget of {max_bytes} bytes "
            f"({len(frame)} rows)"
        )
    return report


__all__ = ["check_frame_memory_limit", "estimate_frame_bytes", "trace_processor_block"]
