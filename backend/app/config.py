"""Central configuration. Everything tunable comes from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass

# ---------------------------------------------------------------- environment
BINANCE_BASE_URL = os.getenv("BINANCE_BASE_URL", "https://api.binance.com/api/v3")
TWELVEDATA_BASE_URL = os.getenv("TWELVEDATA_BASE_URL", "https://api.twelvedata.com")
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY", "")
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "8"))

CORS_ORIGINS = [
    o.strip()
    for o in os.getenv("CORS_ORIGINS", "http://localhost:3100").split(",")
    if o.strip()
]

# ------------------------------------------------------------------ intervals
# Interval label -> length in seconds. The "time limit" of a prediction is
# interval * horizon (e.g. interval="1h", horizon=4 -> "next 4 hours").
INTERVALS: dict[str, int] = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}
INTERVAL_NAMES = tuple(INTERVALS)

MAX_CANDLES = 1000


def cache_ttl_seconds(interval: str) -> int:
    """How long fetched candles are reused before asking the provider again."""
    return max(5, min(300, INTERVALS[interval] // 12))


# --------------------------------------------------------------------- assets
@dataclass(frozen=True)
class Asset:
    symbol: str          # our canonical symbol, e.g. "EURUSD"
    asset_class: str     # "crypto" | "forex" | "gold"
    provider: str        # "binance" | "twelvedata"
    provider_symbol: str  # what that provider expects, e.g. "EUR/USD"
    fee_bps: float       # assumed one-way trading cost (fees/spread), basis points

    @property
    def periods_per_year_days(self) -> int:
        """Trading days per year (crypto is 24/7, FX and gold are not)."""
        return 365 if self.asset_class == "crypto" else 260


# NOTE: fee_bps values are ASSUMPTIONS for backtesting. Replace them with the
# real costs of the venue/broker you intend to use.
_CRYPTO = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
_FOREX = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF"]

ASSETS: dict[str, Asset] = {}
for _s in _CRYPTO:
    ASSETS[_s] = Asset(_s, "crypto", "binance", _s, fee_bps=10.0)
for _s in _FOREX:
    ASSETS[_s] = Asset(_s, "forex", "twelvedata", f"{_s[:3]}/{_s[3:]}", fee_bps=2.0)
ASSETS["XAUUSD"] = Asset("XAUUSD", "gold", "twelvedata", "XAU/USD", fee_bps=4.0)

# Symbols shown by /v2/signals (override with WATCHLIST="BTCUSDT,XAUUSD,...")
WATCHLIST = [
    s.strip().upper()
    for s in os.getenv("WATCHLIST", "BTCUSDT,ETHUSDT,SOLUSDT,EURUSD,GBPUSD,XAUUSD").split(",")
    if s.strip()
]

# Default slippage assumption, basis points per side, on top of fee_bps.
DEFAULT_SLIPPAGE_BPS = float(os.getenv("DEFAULT_SLIPPAGE_BPS", "1.0"))
