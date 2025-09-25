"""Compatibility helpers for working with multiple MLflow releases.

This module centralises optional imports so `qlib.workflow` can run even when
users install only the MLflow core package (without Azure extras) or when
internals move between major versions.
"""

from __future__ import annotations

import logging
from enum import Enum

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ViewType moved across modules between MLflow versions. Fall back to a minimal
# replica if we cannot import it directly.
# ---------------------------------------------------------------------------
try:  # MLflow <= 2.9
    from mlflow.entities import ViewType as _ViewType  # type: ignore[attr-defined]
except (ImportError, AttributeError):
    try:  # MLflow >= 2.10
        from mlflow.entities.view_type import ViewType as _ViewType  # type: ignore[attr-defined]
    except (ImportError, AttributeError):
        _ViewType = None  # type: ignore[assignment]


if _ViewType is None:

    class _FallbackViewType(Enum):
        ACTIVE_ONLY = 1
        DELETED_ONLY = 2
        ALL = 3

    ViewType = _FallbackViewType  # type: ignore[assignment]
else:
    ViewType = _ViewType


# ---------------------------------------------------------------------------
# Azure artifact repository lives behind an extra; keep the import optional.
# ---------------------------------------------------------------------------
try:
    from mlflow.store.artifact.azure_blob_artifact_repo import (  # type: ignore[attr-defined]
        AzureBlobArtifactRepository as _AzureBlobArtifactRepository,
    )
except (ImportError, AttributeError):
    _AzureBlobArtifactRepository = None  # type: ignore[assignment]

AzureBlobArtifactRepository = _AzureBlobArtifactRepository


# ---------------------------------------------------------------------------
# Helper utilities.
# ---------------------------------------------------------------------------


def ensure_min_param_value_limit(min_limit: int = 1000) -> None:
    """Keep MLflow's parameter length cap at or above ``min_limit``.

    Older releases hard-coded a 500-character ceiling which Qlib already
    overrides. Modern MLflow versions default to 6000, so we only patch when
    necessary to avoid shrinking the upstream limit.
    """

    try:
        from mlflow.utils import validation  # type: ignore
    except Exception:  # pragma: no cover - defensive guard for future changes
        logger.debug("mlflow.utils.validation unavailable; skipping param limit tweak")
        return

    current = getattr(validation, "MAX_PARAM_VAL_LENGTH", None)
    if current is None:
        logger.debug("mlflow.utils.validation lacks MAX_PARAM_VAL_LENGTH")
        return

    if current < min_limit:
        validation.MAX_PARAM_VAL_LENGTH = min_limit


def has_azure_blob_artifact_repo() -> bool:
    return AzureBlobArtifactRepository is not None


__all__ = [
    "AzureBlobArtifactRepository",
    "ViewType",
    "ensure_min_param_value_limit",
    "has_azure_blob_artifact_repo",
]
