"""Market-data layer: one function, ``get_candles``, for every asset class.

* Crypto  -> Binance klines (public endpoint, no key)
* Forex / gold -> Twelve Data (needs TWELVEDATA_API_KEY)

Features: symbol validation (no URL injection), a TTL cache so many dashboard
clients cost one upstream request, stale-data fallback when the provider is
down, and removal of the still-forming last candle for ML use.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from typing import Callable

import pandas as pd
import requests

from .. import config
from ..config import Asset

log = logging.getLogger(__name__)

COLUMNS = ["time", "open", "high", "low", "close", "volume"]  # time = open time, ms

_SYMBOL_RE = re.compile(r"^[A-Z0-9]{5,15}$")


class ProviderError(RuntimeError):
    """The upstream data provider failed or returned something unusable."""


class UnknownSymbol(ValueError):
    """Symbol is not supported."""


# --------------------------------------------------------------------- assets
def get_asset(symbol: str) -> Asset:
    """Resolve a user-supplied symbol to an Asset, or raise UnknownSymbol.

    Registered assets are looked up directly. Any other ``*USDT`` pair is
    accepted as Binance crypto (the original API allowed arbitrary pairs), but
    only after a strict character check so it can never alter the request URL.
    """
    s = (symbol or "").strip().upper()
    if s in config.ASSETS:
        return config.ASSETS[s]
    if _SYMBOL_RE.match(s) and s.endswith("USDT"):
        return Asset(s, "crypto", "binance", s, fee_bps=10.0)
    raise UnknownSymbol(f"Unsupported symbol: {symbol!r}")


# ---------------------------------------------------------------------- cache
class _Entry:
    __slots__ = ("df", "stored_at")

    def __init__(self, df: pd.DataFrame, stored_at: float):
        self.df = df
        self.stored_at = stored_at


_cache: dict[tuple, _Entry] = {}
_cache_lock = threading.Lock()
_key_locks: dict[tuple, threading.Lock] = {}


def clear_cache() -> None:
    with _cache_lock:
        _cache.clear()
        _key_locks.clear()


def _lock_for(key: tuple) -> threading.Lock:
    with _cache_lock:
        return _key_locks.setdefault(key, threading.Lock())


# ------------------------------------------------------------------ providers
_session = requests.Session()


def _http_get_json(url: str, params: dict) -> object:
    try:
        r = _session.get(url, params=params, timeout=config.REQUEST_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except (requests.RequestException, ValueError) as exc:
        raise ProviderError(f"request to {url} failed: {exc}") from exc


def _fetch_binance(asset: Asset, interval: str, limit: int) -> pd.DataFrame:
    data = _http_get_json(
        f"{config.BINANCE_BASE_URL}/klines",
        {"symbol": asset.provider_symbol, "interval": interval, "limit": limit},
    )
    if not isinstance(data, list):
        # Binance returns {"code": -1121, "msg": "..."} for bad symbols, and
        # geo-blocked regions get an error object as well.
        raise ProviderError(f"Binance returned an error: {data!r}")
    rows = [[int(c[0]), float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5])] for c in data]
    return pd.DataFrame(rows, columns=COLUMNS)


_TD_INTERVALS = {"1m": "1min", "5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h", "1d": "1day"}
_EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def _fetch_twelvedata(asset: Asset, interval: str, limit: int) -> pd.DataFrame:
    if not config.TWELVEDATA_API_KEY:
        raise ProviderError(
            "TWELVEDATA_API_KEY is not set; forex and gold need a Twelve Data key"
        )
    data = _http_get_json(
        f"{config.TWELVEDATA_BASE_URL}/time_series",
        {
            "symbol": asset.provider_symbol,
            "interval": _TD_INTERVALS[interval],
            "outputsize": limit,
            "timezone": "UTC",
            "order": "ASC",
            "apikey": config.TWELVEDATA_API_KEY,
        },
    )
    if not isinstance(data, dict) or data.get("status") == "error" or "values" not in data:
        msg = data.get("message") if isinstance(data, dict) else data
        raise ProviderError(f"Twelve Data error: {msg}")
    values = data["values"]
    if not values:
        raise ProviderError("Twelve Data returned no candles")
    df = pd.DataFrame(values)
    # Resolution-independent conversion to epoch milliseconds (pandas 3 safe).
    ts = pd.to_datetime(df["datetime"], utc=True)
    df["time"] = ((ts - _EPOCH) // pd.Timedelta(milliseconds=1)).astype("int64")
    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype(float)
    df["volume"] = df["volume"].astype(float) if "volume" in df else 0.0  # FX often has none
    return df[COLUMNS].sort_values("time").reset_index(drop=True)


_PROVIDERS: dict[str, Callable[[Asset, str, int], pd.DataFrame]] = {
    "binance": _fetch_binance,
    "twelvedata": _fetch_twelvedata,
}


def _fetch(asset: Asset, interval: str, limit: int) -> pd.DataFrame:
    return _PROVIDERS[asset.provider](asset, interval, limit)


# ------------------------------------------------------------------ public API
def get_candles(
    symbol: str,
    interval: str = "1m",
    limit: int = 500,
    drop_incomplete: bool = True,
) -> pd.DataFrame:
    """Return OHLCV candles (oldest first) with columns ``COLUMNS``.

    ``drop_incomplete`` removes the candle that is still forming. Models must
    never train or predict on a half-finished bar, so it defaults to True;
    pass False for live charts.
    """
    asset = get_asset(symbol)
    if interval not in config.INTERVALS:
        raise ValueError(f"interval must be one of {list(config.INTERVALS)}")
    limit = max(1, min(int(limit), config.MAX_CANDLES))

    key = (asset.symbol, interval, limit)
    ttl = config.cache_ttl_seconds(interval)

    with _lock_for(key):  # one upstream fetch per key, even under load
        entry = _cache.get(key)
        now = time.time()
        if entry is not None and now - entry.stored_at < ttl:
            df = entry.df
        else:
            try:
                df = _fetch(asset, interval, limit)
                _cache[key] = _Entry(df, now)
            except ProviderError as exc:
                stale_ok = entry is not None and now - entry.stored_at < max(60, 5 * ttl)
                if not stale_ok:
                    raise
                log.warning("provider failed (%s); serving stale %s data", exc, key)
                df = entry.df

    if drop_incomplete:
        cutoff_ms = int(time.time() * 1000) - config.INTERVALS[interval] * 1000
        df = df[df["time"] <= cutoff_ms]
    return df.reset_index(drop=True).copy()
