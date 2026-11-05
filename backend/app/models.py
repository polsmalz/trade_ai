"""Model factory. Deliberately small, regularised models: with noisy financial
data a big model mostly memorises noise. 'persist' is a no-learning baseline
(predict that the next move repeats the last bar) that any real model must beat."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_NAMES = ("logreg", "hgb", "persist")


class ConstantModel:
    """Used when the training window contains only one class."""

    def __init__(self, prob_up: float):
        self.prob_up = float(prob_up)

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.full(n, 1 - self.prob_up), np.full(n, self.prob_up)])


class PersistenceBaseline:
    """Predicts UP if the last bar was up, else DOWN. Needs no training."""

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        up = (pd.DataFrame(X)["ret_1"].to_numpy() > 0).astype(float)
        p = np.where(up > 0, 0.75, 0.25)
        return np.column_stack([1 - p, p])


def make_model(name: str):
    if name == "logreg":
        return Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=0.3, max_iter=1000)),
        ])
    if name == "hgb":
        return HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=150,
            l2_regularization=1.0, min_samples_leaf=30,
            early_stopping=False, random_state=0,
        )
    if name == "persist":
        return PersistenceBaseline()
    raise ValueError(f"model must be one of {list(MODEL_NAMES)}")


def fit_model(name: str, X: pd.DataFrame, y: np.ndarray):
    """Fit ``name`` on (X, y); falls back to a constant model for 1-class data."""
    if len(np.unique(y)) < 2:
        return ConstantModel(float(np.mean(y)) if len(y) else 0.5)
    model = make_model(name)
    model.fit(X, y)
    return model


def prob_up(model, X: pd.DataFrame) -> np.ndarray:
    """Probability that price is higher at the horizon, for each row of X."""
    proba = model.predict_proba(X)
    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "named_steps"):
        classes = model.named_steps["clf"].classes_
    if classes is not None and 1 in list(classes):
        return proba[:, list(classes).index(1)]
    return proba[:, -1]
