"""Live prediction service: trains on recent history and caches the model per
(symbol, interval, horizon, model, last-bar). A new bar means a new cache key,
so the model automatically retrains as the market moves (the original code
trained once and never again)."""
from __future__ import annotations

import threading
from collections import OrderedDict

import numpy as np

from . import config
from .data.providers import get_asset, get_candles
from .features import WARMUP, build_features, forward_log_return, rsi
from .models import fit_model, prob_up

DISCLAIMER = (
    "Experimental model output, not financial advice. Check /v2/backtest for "
    "out-of-sample evidence; past performance does not guarantee future results."
)

_MODEL_CACHE: "OrderedDict[tuple, object]" = OrderedDict()
_MODEL_CACHE_MAX = 64
_lock = threading.Lock()


def horizon_label(interval: str, horizon: int) -> str:
    seconds = config.INTERVALS[interval] * horizon
    if seconds % 86400 == 0:
        return f"{seconds // 86400}d"
    if seconds % 3600 == 0:
        return f"{seconds // 3600}h"
    return f"{seconds // 60}m"


def rsi_signal(value: float | None) -> str:
    """Classic RSI rule, used for the simple 'signal' column on the dashboard."""
    if value is None or np.isnan(value):
        return "HOLD"
    if value < 30:
        return "BUY"
    if value > 70:
        return "SELL"
    return "HOLD"


def predict(symbol: str, interval: str = "1h", horizon: int = 1, model_name: str = "hgb",
            candles: int = 1000) -> dict:
    asset = get_asset(symbol)
    df = get_candles(asset.symbol, interval, candles, drop_incomplete=True)
    if len(df) < WARMUP + 60 + horizon:
        raise ValueError(f"not enough candles ({len(df)}) to train for this interval/horizon")

    feat = build_features(df)
    fwd = forward_log_return(df, horizon)
    last_time = int(df["time"].iloc[-1])

    key = (asset.symbol, interval, horizon, model_name, last_time)
    with _lock:
        model = _MODEL_CACHE.get(key)
        if model is not None:
            _MODEL_CACHE.move_to_end(key)
    if model is None:
        usable = feat.notna().all(axis=1) & fwd.notna()
        X, y = feat[usable], (fwd[usable] > 0).astype(int).to_numpy()
        model = fit_model(model_name, X, y)
        with _lock:
            _MODEL_CACHE[key] = model
            while len(_MODEL_CACHE) > _MODEL_CACHE_MAX:
                _MODEL_CACHE.popitem(last=False)

    latest = feat.iloc[[-1]]
    if latest.isna().any(axis=None):
        raise ValueError("latest bar has undefined features (not enough history)")
    p = float(prob_up(model, latest)[0])
    return {
        "symbol": asset.symbol,
        "asset_class": asset.asset_class,
        "interval": interval,
        "horizon_bars": horizon,
        "horizon": horizon_label(interval, horizon),
        "model": model_name,
        "prediction": "UP" if p >= 0.5 else "DOWN",
        "prob_up": round(p, 4),
        "confidence": round(max(p, 1 - p) * 100, 2),   # confidence in the CALLED direction
        "price": float(df["close"].iloc[-1]),
        "as_of_ms": last_time,
        "disclaimer": DISCLAIMER,
    }


def snapshot(symbol: str, interval: str, horizon: int, model_name: str) -> dict:
    """Price + RSI rule signal + model prediction, for the watchlist table."""
    pred = predict(symbol, interval, horizon, model_name)
    df = get_candles(symbol, interval, 100, drop_incomplete=True)
    r = float(rsi(df["close"]).iloc[-1])
    return {
        "symbol": pred["symbol"],
        "asset_class": pred["asset_class"],
        "price": pred["price"],
        "rsi": round(r, 2),
        "signal": rsi_signal(r),
        "prediction": pred["prediction"],
        "confidence": pred["confidence"],
        "horizon": pred["horizon"],
    }
