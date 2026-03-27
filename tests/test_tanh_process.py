# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for TanhProcess fix (Fixes #1687).

TanhProcess uses MultiIndex columns where level 0 contains group names
(e.g. LABEL0, FEATURE0) and level 1 contains individual feature names.
The mask must use get_level_values(0) to identify LABEL columns correctly.
"""

import numpy as np
import pandas as pd
import pytest


def test_tanh_process_leaves_labels_untouched():
    """TanhProcess should apply tanh only to non-LABEL columns."""
    from qlib.data.dataset.processor import TanhProcess

    # Build a DataFrame with MultiIndex columns:
    #   level-0 (group): LABEL0, FEATURE0, FEATURE0
    #   level-1 (name):  LABEL0, FEATURE0, FEATURE1
    columns = pd.MultiIndex.from_tuples(
        [("LABEL0", "LABEL0"), ("FEATURE0", "FEATURE0"), ("FEATURE0", "FEATURE1")],
        names=["group", "feature"],
    )
    index = pd.MultiIndex.from_tuples(
        [("2023-01-01", "STOCK_A"), ("2023-01-01", "STOCK_B")],
        names=["datetime", "instrument"],
    )
    data = np.array([[0.5, 2.0, 3.0], [0.8, 4.0, 5.0]])
    df = pd.DataFrame(data, index=index, columns=columns)
    label_before = df[("LABEL0", "LABEL0")].copy()

    proc = TanhProcess()
    result = proc(df)

    # LABEL columns must be unchanged
    pd.testing.assert_series_equal(result[("LABEL0", "LABEL0")], label_before)

    # FEATURE columns should have tanh(x - 1) applied
    expected_f0 = np.tanh(np.array([2.0, 4.0]) - 1)
    np.testing.assert_allclose(
        result[("FEATURE0", "FEATURE0")].values, expected_f0, atol=1e-7
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
