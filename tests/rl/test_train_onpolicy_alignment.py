# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Tests for data-alignment validation in qlib.rl.contrib.train_onpolicy.

Regression test for GitHub issue #1972:
  https://github.com/microsoft/qlib/issues/1972

Previously, misaligned ``default_start_time_index`` / ``default_end_time_index``
values triggered a bare ``AssertionError`` with no message.  The fix extracts
the validation into ``_validate_time_alignment`` and raises a descriptive
``ValueError`` so that users immediately understand what went wrong and how to
correct their config.

These tests import only ``_validate_time_alignment``, which is a plain-Python
function with zero heavy dependencies, so they run without tianshou/torch/gym
being installed.
"""

import pytest

# ---------------------------------------------------------------------------
# Import the standalone validation helper directly from the lightweight
# _utils module, which has **zero** heavy dependencies (no torch/tianshou/gym).
# ---------------------------------------------------------------------------
from qlib.rl.contrib._utils import validate_time_alignment


# ---------------------------------------------------------------------------
# Tests: valid configurations should NOT raise
# ---------------------------------------------------------------------------

class TestAlignmentValid:
    """Valid index/granularity combinations should not raise."""

    def test_granularity_1_start_0_end_240(self):
        """Any index is valid when data_granularity=1 (default)."""
        validate_time_alignment(0, 240, 1)  # should not raise

    def test_granularity_5_start_0_end_235(self):
        """0 and 235 are both multiples of 5."""
        validate_time_alignment(0, 235, 5)

    def test_granularity_30_start_0_end_210(self):
        """0 and 210 are both multiples of 30."""
        validate_time_alignment(0, 210, 30)

    def test_granularity_10_start_10_end_240(self):
        """10 and 240 are both multiples of 10."""
        validate_time_alignment(10, 240, 10)

    def test_granularity_2_start_0_end_240(self):
        """0 and 240 are both multiples of 2."""
        validate_time_alignment(0, 240, 2)


# ---------------------------------------------------------------------------
# Tests: invalid configurations MUST raise ValueError (not AssertionError)
# ---------------------------------------------------------------------------

class TestAlignmentInvalid:
    """Misaligned indices should raise a descriptive ValueError, not AssertionError."""

    def test_start_not_aligned_raises_value_error(self):
        """start_time_index=1 with data_granularity=5 is invalid."""
        with pytest.raises(ValueError, match="default_start_time_index"):
            validate_time_alignment(1, 235, 5)

    def test_end_not_aligned_raises_value_error(self):
        """end_time_index=237 with data_granularity=5 is invalid."""
        with pytest.raises(ValueError, match="default_end_time_index"):
            validate_time_alignment(0, 237, 5)

    def test_error_message_contains_actual_start_value(self):
        """The error message must include the offending start value and granularity."""
        with pytest.raises(ValueError) as exc_info:
            validate_time_alignment(3, 240, 10)
        msg = str(exc_info.value)
        assert "3" in msg, "Error message should include the bad start index"
        assert "10" in msg, "Error message should include data_granularity"

    def test_error_message_contains_hint(self):
        """The error message must contain a suggested corrected value."""
        with pytest.raises(ValueError) as exc_info:
            validate_time_alignment(3, 240, 5)
        msg = str(exc_info.value)
        # Nearest lower multiple of 5 for start=3 is 0
        assert "0" in msg, "Error message should suggest 0 as a valid start index"

    def test_raises_value_error_not_assertion_error(self):
        """The old code raised AssertionError; the new code must raise ValueError.

        This is the key regression check for issue #1972.
        """
        with pytest.raises(ValueError):
            validate_time_alignment(1, 240, 2)
        # Also ensure it is NOT accidentally an AssertionError
        try:
            validate_time_alignment(1, 240, 2)
        except AssertionError:
            pytest.fail(
                "Raised AssertionError instead of ValueError — regression of issue #1972!"
            )
        except ValueError:
            pass  # expected

    def test_granularity_30_start_1(self):
        """start=1 is not a multiple of 30."""
        with pytest.raises(ValueError, match="default_start_time_index"):
            validate_time_alignment(1, 210, 30)

    def test_both_misaligned_raises_on_start_first(self):
        """When both indices are misaligned, start error is raised first."""
        with pytest.raises(ValueError, match="default_start_time_index"):
            validate_time_alignment(3, 238, 5)

    def test_only_end_misaligned(self):
        """A valid start but invalid end should still raise on end."""
        with pytest.raises(ValueError, match="default_end_time_index"):
            validate_time_alignment(0, 238, 5)

    def test_error_message_contains_modulo_result(self):
        """The error message should show the actual remainder."""
        with pytest.raises(ValueError) as exc_info:
            validate_time_alignment(3, 240, 5)
        msg = str(exc_info.value)
        # 3 % 5 = 3 — the remainder should appear in the message
        assert "3" in msg
