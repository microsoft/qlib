#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.
"""Lightweight contrib.workflow package init.

Avoid importing heavy submodules at import time to prevent unintended
side-effects and circular imports when users import a specific submodule
like `qlib.contrib.workflow.crypto_record_temp`.
"""

from __future__ import annotations

import importlib
from typing import Any, TYPE_CHECKING

__all__ = ["MultiSegRecord", "SignalMseRecord"]

if TYPE_CHECKING:  # only for type checkers; no runtime import
    from .record_temp import MultiSegRecord, SignalMseRecord  # noqa: F401


def __getattr__(name: str) -> Any:
    if name in ("MultiSegRecord", "SignalMseRecord"):
        mod = importlib.import_module(__name__ + ".record_temp")
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
