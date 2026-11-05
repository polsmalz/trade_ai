"""FastAPI app.

* v1 endpoints (/, /signals, /predict/{symbol}, /chart/{symbol}) are kept so the
  current dashboard keeps working, but now run on the cached data layer and no
  longer crash when the exchange API fails.
* v2 endpoints live under /v2 (see app/api_v2.py).
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app import config, legacy, service
from app.api_v2 import router as v2_router
from app.data.providers import ProviderError, UnknownSymbol, get_candles
from app.features import rsi

app = FastAPI(title="Trade AI v2 API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,  # was ["*"] together with credentials (unsafe)
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)
app.include_router(v2_router)

LEGACY_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


def _candles_or_http_error(symbol: str, interval: str, limit: int, drop_incomplete: bool):
    try:
        return get_candles(symbol, interval, limit, drop_incomplete)
    except UnknownSymbol as exc:
        raise HTTPException(404, str(exc))
    except ProviderError as exc:
        raise HTTPException(502, f"data provider error: {exc}")


@app.get("/")
def home():
    return {"message": "AI Trading System Running", "version": app.version}


# ------------------------------------------------------------- v1 (kept alive)
@app.get("/signals")
def multi_signals():
    results = []
    for symbol in LEGACY_SYMBOLS:
        try:
            df = get_candles(symbol, "1m", 100, drop_incomplete=False)
            r = float(rsi(df["close"]).iloc[-1])
            results.append({
                "symbol": symbol,
                "price": float(df["close"].iloc[-1]),
                "rsi": round(r, 2),
                "signal": service.rsi_signal(r),
            })
        except (ProviderError, UnknownSymbol) as exc:
            results.append({"symbol": symbol, "price": None, "rsi": None, "signal": "N/A", "error": str(exc)})
    return results


@app.get("/predict/{symbol}")
def predict(symbol: str):
    df = _candles_or_http_error(symbol, "1m", 100, drop_incomplete=True)
    result = legacy.predict_price_direction(df["close"].tolist())
    return {
        "symbol": symbol.upper(),
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "prob_up": result["prob_up"],
        "note": "legacy raw-price model; use /v2/predict/{symbol}",
    }


@app.get("/chart/{symbol}")
def chart(symbol: str):
    df = _candles_or_http_error(symbol, "1m", 50, drop_incomplete=False)
    return [
        {"time": int(r.time), "open": r.open, "high": r.high, "low": r.low, "close": r.close}
        for r in df.itertuples()
    ]
