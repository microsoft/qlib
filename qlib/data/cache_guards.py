# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Hardened handler-cache helpers for ``examples/data_demo/data_cache_demo.py``.

The demo round-trips a ``DataHandlerLP`` pickle through ``file://`` URIs. These
helpers add the missing production guards without changing the demo flow:
``file://`` allow-listing, base-directory confinement (reuses
:func:`qlib.backtest.validation.sanitize_cache_path`), size budgets, and an
explicit ``trusted`` opt-in for unpickling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

from qlib.backtest.validation import sanitize_cache_path

DEFAULT_MAX_CACHE_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB per handler pickle


def resolve_cache_file_uri(uri: str, base_dir: Union[str, Path]) -> Path:
    """Resolve a ``file://`` handler-cache URI strictly inside ``base_dir``.

    Raises
    ------
    ValueError
        If the scheme is not ``file://`` or the target escapes ``base_dir``.
    """
    if not isinstance(uri, str) or not uri.startswith("file://"):
        raise ValueError(f"only file:// cache URIs are supported, got {uri!r}")
    return sanitize_cache_path(base_dir, uri[len("file://") :])


def check_cache_file_budget(path: Union[str, Path], *, max_bytes: int = DEFAULT_MAX_CACHE_BYTES) -> int:
    """Return file size after enforcing an explicit cache budget.

    Raises
    ------
    FileNotFoundError
        If the cache file does not exist.
    MemoryError
        If the file exceeds ``max_bytes``.
    """
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"handler cache not found: {resolved}")
    size = resolved.stat().st_size
    if size > max_bytes:
        raise MemoryError(f"handler cache {resolved} uses ~{size} bytes, exceeding budget of {max_bytes} bytes")
    return size


def require_trusted_cache_load(trusted: bool, path: Union[str, Path]) -> None:
    """Require explicit opt-in before unpickling a handler cache file."""
    if not trusted:
        raise ValueError(
            f"refusing to unpickle handler cache {path}: pass trusted=True only for caches you produced"
        )


__all__ = [
    "DEFAULT_MAX_CACHE_BYTES",
    "check_cache_file_budget",
    "require_trusted_cache_load",
    "resolve_cache_file_uri",
]
