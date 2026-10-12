# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Lightweight validation utilities for RL training configuration.

This module intentionally has **zero** heavy dependencies (no torch, tianshou,
gym, etc.) so that the helpers here can be imported and unit-tested in any
environment, including lightweight CI images.
"""

from __future__ import annotations


def validate_time_alignment(
    default_start_time_index: int,
    default_end_time_index: int,
    data_granularity: int,
) -> None:
    """Validate that start/end time indices are aligned to ``data_granularity``.

    Both ``default_start_time_index`` and ``default_end_time_index`` must be
    exact multiples of ``data_granularity``.  This is required because the RL
    environment slices each trading day into uniform windows of
    ``data_granularity`` ticks; a misaligned start/end index would silently
    corrupt step boundaries.

    Parameters
    ----------
    default_start_time_index : int
        The starting tick index, taken from
        ``data_config["source"]["default_start_time_index"]``.
    default_end_time_index : int
        The ending tick index (exclusive), taken from
        ``data_config["source"]["default_end_time_index"]``.
    data_granularity : int
        Number of raw ticks grouped into a single data point (from
        ``simulator_config["data_granularity"]``).

    Raises
    ------
    ValueError
        If either index is not divisible by ``data_granularity``.

    Examples
    --------
    Valid – 0 and 240 are both multiples of 5:

    >>> validate_time_alignment(0, 240, 5)

    Invalid – 3 is not a multiple of 5:

    >>> validate_time_alignment(3, 240, 5)  # doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    ValueError: data_config['source']['default_start_time_index'] (=3) ...
    """
    if default_start_time_index % data_granularity != 0:
        _suggested = (default_start_time_index // data_granularity) * data_granularity
        raise ValueError(
            f"data_config['source']['default_start_time_index'] "
            f"(={default_start_time_index}) must be divisible by "
            f"data_granularity (={data_granularity}), but "
            f"{default_start_time_index} % {data_granularity} = "
            f"{default_start_time_index % data_granularity}. "
            f"Hint: set default_start_time_index to a multiple of "
            f"{data_granularity} (e.g. {_suggested})."
        )

    if default_end_time_index % data_granularity != 0:
        _suggested_end = (default_end_time_index // data_granularity) * data_granularity
        raise ValueError(
            f"data_config['source']['default_end_time_index'] "
            f"(={default_end_time_index}) must be divisible by "
            f"data_granularity (={data_granularity}), but "
            f"{default_end_time_index} % {data_granularity} = "
            f"{default_end_time_index % data_granularity}. "
            f"Hint: set default_end_time_index to a multiple of "
            f"{data_granularity} (e.g. {_suggested_end})."
        )
