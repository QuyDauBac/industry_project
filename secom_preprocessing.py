import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class MissingRateSelector(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.70):
        self.threshold = threshold

    def fit(self, X, y=None):
        if not 0 <= self.threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")

        input_values = self._to_array(X)
        if input_values.shape[0] == 0 or input_values.shape[1] == 0:
            raise ValueError("X must contain at least one row and one feature")

        self.n_features_in_ = input_values.shape[1]
        self.missing_rates_ = np.isnan(input_values).mean(axis=0)
        self.support_ = self.missing_rates_ <= self.threshold
        if not self.support_.any():
            raise ValueError("The missing-rate threshold removed every feature")
        return self

    def transform(self, X):
        input_values = self._to_array(X)
        if input_values.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {input_values.shape[1]} features; expected {self.n_features_in_}"
            )
        return input_values[:, self.support_]

    def get_support(self, indices=False):
        if indices:
            return np.flatnonzero(self.support_)
        return self.support_.copy()

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = np.asarray(
                [f"x{i}" for i in range(self.n_features_in_)], dtype=object
            )
        else:
            input_features = np.asarray(input_features, dtype=object)
        if input_features.shape[0] != self.n_features_in_:
            raise ValueError("input_features has a different length than the fitted data")
        return input_features[self.support_]

    @staticmethod
    def _to_array(X):
        if hasattr(X, "to_numpy"):
            input_values = X.to_numpy(dtype=float)
        else:
            input_values = np.asarray(X, dtype=float)
        if input_values.ndim != 2:
            raise ValueError("X must be a two-dimensional array")
        return input_values