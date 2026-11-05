"""The original v1 behaviour (logistic regression on 10 raw 1-minute closes),
kept so the existing dashboard keeps working while v2 is built alongside.

Known weakness (documented, not fixed): raw price levels are non-stationary, so
this model has essentially no predictive value. Use /v2/predict instead. Fixes
applied: confidence now reflects the called direction, the pickle is loaded
defensively, and failures no longer crash the endpoint."""
from __future__ import annotations

import logging
import os

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

log = logging.getLogger(__name__)

MODEL_PATH = os.getenv("LEGACY_MODEL_PATH", "model.pkl")
WINDOW = 10


def create_dataset(prices, window: int = WINDOW):
    X, y = [], []
    for i in range(len(prices) - window - 1):
        X.append(prices[i:i + window])
        y.append(1 if prices[i + window] > prices[i + window - 1] else 0)
    return np.array(X), np.array(y)


def train_model(prices):
    X, y = create_dataset(prices)
    if len(X) == 0 or len(np.unique(y)) < 2:
        return None
    model = LogisticRegression()
    model.fit(X, y)
    try:
        joblib.dump(model, MODEL_PATH)
    except OSError as exc:
        log.warning("could not save %s: %s", MODEL_PATH, exc)
    return model


def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception as exc:  # pickles break across scikit-learn versions
        log.warning("could not load %s (%s); will retrain", MODEL_PATH, exc)
        return None


def predict_price_direction(prices) -> dict:
    if len(prices) < WINDOW + 2:
        return {"prediction": "HOLD", "prob_up": 0.5, "confidence": 0}

    model = load_model()
    if model is None:
        model = train_model(prices)
    if model is None:
        return {"prediction": "HOLD", "prob_up": 0.5, "confidence": 0}

    X = np.array(prices[-WINDOW:]).reshape(1, -1)
    try:
        p = float(model.predict_proba(X)[0][1])
    except Exception as exc:
        log.warning("legacy model failed (%s)", exc)
        p = 0.5
    return {
        "prediction": "UP" if p > 0.5 else "DOWN",
        "prob_up": round(p, 4),
        "confidence": round(max(p, 1 - p) * 100, 2),
    }
