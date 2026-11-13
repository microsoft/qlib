import unittest

import numpy as np
import pandas as pd

from qlib.contrib.model.double_ensemble import DEnsembleModel


class _PermutationSensitiveModel:
    """Predicts NaN when the column ``col`` differs from the training data, else 0."""

    def __init__(self, x, col):
        self.x = x
        self.col = col

    def predict(self, x):
        if np.array_equal(x[:, self.col], self.x[:, self.col]):
            return np.zeros(len(x))
        return np.full(len(x), np.nan)


class TestDEnsembleFeatureSelection(unittest.TestCase):
    def test_nan_g_value_is_treated_as_zero(self):
        np.random.seed(0)
        n, n_feat = 50, 5
        features = [f"F{i}" for i in range(n_feat)]
        x = pd.DataFrame(np.arange(n * n_feat, dtype=float).reshape(n, n_feat), columns=features)
        y = pd.DataFrame({"LABEL0": np.linspace(-1, 1, n)})
        df_train = pd.concat({"feature": x, "label": y}, axis=1)

        model = DEnsembleModel(num_models=1, bins_fs=5)
        # shuffling F1 makes the prediction NaN, so its g-value is NaN
        model.ensemble = [_PermutationSensitiveModel(x.values, col=1)]
        model.sub_features = [pd.Index(features)]
        loss_values = model.get_loss(y.values.squeeze(), np.zeros(n))

        res_feat = model.feature_selection(df_train, loss_values)

        # all g-values end up 0, so every feature lands in one bin and 80% of them are kept
        self.assertEqual(len(res_feat), 4)
        self.assertTrue(set(res_feat) <= set(features))


if __name__ == "__main__":
    unittest.main()
