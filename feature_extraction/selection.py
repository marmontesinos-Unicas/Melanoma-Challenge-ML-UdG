"""Small sklearn transformers for feature selection (usable inside a Pipeline)."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """
    Drop features that are almost a copy of an earlier one (|Pearson r| > threshold).
    Fitted on the training fold only, so it is safe inside cross-validation.
    """

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        sd = X.std(0)
        Z = (X - X.mean(0)) / np.where(sd > 0, sd, 1)
        corr = np.abs(Z.T @ Z) / len(X)
        upper = np.triu(corr, 1)
        self.keep_ = ~(upper > self.threshold).any(axis=0)
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.keep_]

    def get_support(self):
        return self.keep_
