"""Feature engineering. Every feature at row t uses data up to and including
row t only (rolling/ewm windows looking backwards), so there is no look-ahead.
Features are scale-free (returns, ratios, z-scores) rather than raw prices,
which is what the original raw-price model got wrong."""
from __future__ import annotations

import numpy as np
import pandas as pd

WARMUP = 50  # rows needed before every feature is defined

FEATURE_COLUMNS = [
    "ret_1", "ret_2", "ret_3", "ret_4", "ret_5",
    "mom_5", "mom_10", "mom_20",
    "vol_10", "vol_20",
    "rsi_14", "macd_hist", "bb_z", "atr_14", "hl_range",
    "vol_z", "dist_sma_50",
]


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI, 0-100. Flat series gives 50."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    total = gain + loss
    out = 100 * gain / total
    return out.where(total != 0, 50.0)  # flat market -> neutral 50; warmup stays NaN


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame (same index as ``df``) with FEATURE_COLUMNS.
    Leading rows are NaN until the longest window fills."""
    close, high, low = df["close"], df["high"], df["low"]
    lr = np.log(close).diff()

    f = pd.DataFrame(index=df.index)
    for k in range(1, 6):
        f[f"ret_{k}"] = lr.shift(k - 1)
    for n in (5, 10, 20):
        f[f"mom_{n}"] = np.log(close / close.shift(n))
    f["vol_10"] = lr.rolling(10).std()
    f["vol_20"] = lr.rolling(20).std()

    f["rsi_14"] = rsi(close, 14)

    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    f["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / close

    ma20, sd20 = close.rolling(20).mean(), close.rolling(20).std()
    f["bb_z"] = ((close - ma20) / sd20.where(sd20 > 0)).fillna(0.0).where(ma20.notna())

    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    f["atr_14"] = tr.rolling(14).mean() / close
    f["hl_range"] = (high - low) / close

    vol = df["volume"]
    if (vol > 0).any():
        vm, vs = vol.rolling(20).mean(), vol.rolling(20).std()
        f["vol_z"] = ((vol - vm) / vs.where(vs > 0)).fillna(0.0).where(vm.notna())
    else:  # forex feeds often carry no volume
        f["vol_z"] = 0.0

    f["dist_sma_50"] = close / close.rolling(50).mean() - 1
    return f[FEATURE_COLUMNS]


def forward_log_return(df: pd.DataFrame, horizon: int) -> pd.Series:
    """log(close[t+h] / close[t]); NaN for the last ``horizon`` rows (unknown yet).
    This is the prediction target (sign) and the realised outcome of a trade."""
    close = df["close"]
    return np.log(close.shift(-horizon) / close)
