# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Explicit payload / memory budgets for distributed rolling-task dispatch.

``examples/model_rolling/task_manager_rolling.py`` fans tasks out to MongoDB and
multiprocess workers with no cap; oversized task dicts cause OOMs and giant
Mongo documents. These helpers give producers one place to enforce limits while
keeping every existing call site working (defaults are permissive).
"""

from __future__ import annotations

import pickle
from typing import Any, Dict, List

DEFAULT_MAX_TASK_BYTES = 4 * 1024 * 1024  # 4 MiB per serialized task
DEFAULT_MAX_TASKS_PER_BATCH = 512


def estimate_task_bytes(task: Any) -> int:
    """Best-effort serialized size of one task (pickle length, 0 on failure)."""
    try:
        return len(pickle.dumps(task))
    except Exception:  # pragma: no cover - unpicklable task detail
        return 0


def check_task_batch_limits(
    tasks: List[Any],
    *,
    max_task_bytes: int = DEFAULT_MAX_TASK_BYTES,
    max_tasks: int = DEFAULT_MAX_TASKS_PER_BATCH,
) -> Dict[str, Any]:
    """Validate a rolling-task batch against explicit payload/memory limits.

    Raises
    ------
    ValueError
        If the batch is empty, exceeds ``max_tasks``, or any task exceeds
        ``max_task_bytes``.
    """
    if max_tasks <= 0:
        raise ValueError("max_tasks must be positive")
    if max_task_bytes <= 0:
        raise ValueError("max_task_bytes must be positive")
    if not tasks:
        raise ValueError("task batch must not be empty")
    if len(tasks) > max_tasks:
        raise ValueError(f"task batch of {len(tasks)} exceeds limit of {max_tasks}")
    sizes = [estimate_task_bytes(task) for task in tasks]
    worst = max(sizes) if sizes else 0
    if worst > max_task_bytes:
        raise ValueError(f"largest task uses ~{worst} bytes, exceeding budget of {max_task_bytes} bytes")
    return {
        "num_tasks": len(tasks),
        "max_task_bytes": worst,
        "total_bytes": sum(sizes),
        "limit_bytes": max_task_bytes,
        "limit_tasks": max_tasks,
    }


def cap_task_batch(
    tasks: List[Any],
    max_tasks: int = DEFAULT_MAX_TASKS_PER_BATCH,
) -> List[Any]:
    """Truncate a batch to ``max_tasks`` (oldest-first, non-destructive)."""
    if max_tasks <= 0:
        raise ValueError("max_tasks must be positive")
    return list(tasks[:max_tasks])


__all__ = [
    "DEFAULT_MAX_TASKS_PER_BATCH",
    "DEFAULT_MAX_TASK_BYTES",
    "cap_task_batch",
    "check_task_batch_limits",
    "estimate_task_bytes",
]
