import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def synthetic_candles(n=1500, seed=0, phi=0.0, start_ms=1_600_000_000_000, step_ms=3_600_000,
                      sigma=0.002, volume=True):
    """Random-walk OHLCV. ``phi`` > 0 adds return autocorrelation (a real,
    learnable signal); phi = 0 is a pure random walk (no signal exists)."""
    rng = np.random.default_rng(seed)
    eps = rng.normal(0, sigma, n)
    r = np.zeros(n)
    for t in range(1, n):
        r[t] = phi * r[t - 1] + eps[t]
    close = 100 * np.exp(np.cumsum(r))
    open_ = np.concatenate([[100.0], close[:-1]])
    spread = np.abs(rng.normal(0, sigma / 2, n)) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    vol = rng.lognormal(3, 0.4, n) if volume else np.zeros(n)
    return pd.DataFrame({
        "time": start_ms + np.arange(n) * step_ms,
        "open": open_, "high": high, "low": low, "close": close, "volume": vol,
    })


@pytest.fixture
def candles():
    return synthetic_candles


@pytest.fixture(autouse=True)
def _clean_state():
    from app.data import providers
    from app import service
    providers.clear_cache()
    service._MODEL_CACHE.clear()
    yield
    providers.clear_cache()
    service._MODEL_CACHE.clear()
