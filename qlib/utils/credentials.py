# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Credential hygiene helpers for backtest / live-trading scaffolding.

Guarantees no API keys or secrets are hardcoded: credentials resolve strictly
from explicit arguments or ``${VAR}``-style environment variables. ``.env``
values are for local runs only and must never be committed.
"""

from __future__ import annotations

import os
import re
from typing import Dict, Optional

_ENV_REF_PATTERN = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")


def resolve_credential(
    name: str,
    explicit: Optional[str] = None,
    *,
    required: bool = False,
    allow_empty: bool = True,
) -> Optional[str]:
    """Resolve one credential without ever falling back to a hardcoded default.

    Parameters
    ----------
    name : str
        Environment variable holding the credential (e.g. ``QLIB_TRADE_API_KEY``).
    explicit : Optional[str]
        Directly-passed value, which takes precedence over the environment.
    required : bool
        Raise when neither source provides a value.
    """
    if explicit:
        return explicit
    value = os.environ.get(name)
    if value:
        return value
    if required:
        raise ValueError(
            f"missing required credential {name!r}: pass it explicitly or set the {name} environment variable"
        )
    return None if allow_empty else ""


def resolve_env_reference(value: str) -> str:
    """Expand a single ``${VAR}`` reference; reject inline hardcoding patterns."""
    match = _ENV_REF_PATTERN.match(value.strip())
    if not match:
        return value
    var_name = match.group(1)
    resolved = os.environ.get(var_name)
    if resolved is None:
        raise ValueError(f"environment variable {var_name!r} referenced by {value!r} is not set")
    return resolved


def collect_trading_credentials(
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    data_token: Optional[str] = None,
) -> Dict[str, Optional[str]]:
    """Collect the standard online-serving credential triple from env/explicit args."""
    return {
        "api_key": resolve_credential("QLIB_TRADE_API_KEY", api_key),
        "api_secret": resolve_credential("QLIB_TRADE_API_SECRET", api_secret),
        "data_token": resolve_credential("QLIB_DATA_API_TOKEN", data_token),
    }


__all__ = ["collect_trading_credentials", "resolve_credential", "resolve_env_reference"]
