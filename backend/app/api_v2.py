"""v2 endpoints: multi-asset, multi-horizon predictions and honest backtests."""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException, Query

from . import config, service
from .backtest import run_backtest
from .data.providers import ProviderError, UnknownSymbol, get_asset, get_candles
from .models import MODEL_NAMES

router = APIRouter(prefix="/v2", tags=["v2"])

Interval = Literal["1m", "5m", "15m", "1h", "4h", "1d"]
ModelName = Literal["logreg", "hgb", "persist"]


def _guard(fn, *args, **kwargs):
    """Map domain errors to proper HTTP status codes instead of a blanket 500."""
    try:
        return fn(*args, **kwargs)
    except UnknownSymbol as exc:
        raise HTTPException(404, str(exc))
    except ProviderError as exc:
        raise HTTPException(502, f"data provider error: {exc}")
    except ValueError as exc:
        raise HTTPException(422, str(exc))


@router.get("/symbols")
def symbols():
    return [
        {
            "symbol": a.symbol,
            "asset_class": a.asset_class,
            "provider": a.provider,
            "available": a.provider != "twelvedata" or bool(config.TWELVEDATA_API_KEY),
            "assumed_fee_bps": a.fee_bps,
        }
        for a in config.ASSETS.values()
    ]


@router.get("/candles/{symbol}")
def candles(symbol: str, interval: Interval = "1h", limit: int = Query(200, ge=10, le=config.MAX_CANDLES)):
    df = _guard(get_candles, symbol, interval, limit, False)
    return df.to_dict("records")


@router.get("/predict/{symbol}")
def predict(
    symbol: str,
    interval: Interval = "1h",
    horizon: int = Query(1, ge=1, le=48, description="How many bars ahead (time limit = interval x horizon)"),
    model: ModelName = "hgb",
):
    return _guard(service.predict, symbol, interval, horizon, model)


@router.get("/signals")
def signals(interval: Interval = "1h", horizon: int = Query(1, ge=1, le=48), model: ModelName = "hgb"):
    """Watchlist table. One failing symbol (e.g. forex without an API key) does
    not break the rest."""
    out = []
    for sym in config.WATCHLIST:
        try:
            out.append(service.snapshot(sym, interval, horizon, model))
        except UnknownSymbol as exc:
            out.append({"symbol": sym, "error": str(exc)})
        except (ProviderError, ValueError) as exc:
            out.append({"symbol": sym, "error": str(exc)})
    return out


@router.get("/backtest/{symbol}")
def backtest(
    symbol: str,
    interval: Interval = "1h",
    horizon: int = Query(1, ge=1, le=48),
    model: ModelName = "hgb",
    threshold: float = Query(0.55, ge=0.5, le=0.99, description="Minimum confidence to take a trade"),
    min_train: int = Query(300, ge=100, le=800),
    step: int = Query(50, ge=5, le=500, description="Retrain every N bars"),
    limit: int = Query(1000, ge=300, le=config.MAX_CANDLES),
    allow_short: bool = True,
    slippage_bps: float = Query(config.DEFAULT_SLIPPAGE_BPS, ge=0, le=100),
    target_hit_rate: float = Query(0.80, ge=0.5, le=0.99),
    compare_baseline: bool = True,
):
    """Walk-forward, out-of-sample test with fees. ``meets_target`` is true only
    if the *lower* 95% confidence bound of the trade hit-rate clears the target
    on at least 200 trades."""

    def run():
        asset = get_asset(symbol)
        df = get_candles(asset.symbol, interval, limit, drop_incomplete=True)
        common = dict(interval=interval, horizon=horizon, asset=asset, threshold=threshold,
                      min_train=min_train, step=step, slippage_bps=slippage_bps,
                      allow_short=allow_short, target_hit_rate=target_hit_rate)
        result = {
            "symbol": asset.symbol, "interval": interval, "horizon": service.horizon_label(interval, horizon),
            "model": model, "candles_used": len(df),
            "results": run_backtest(df, model_name=model, **common),
        }
        if compare_baseline and model != "persist":
            result["baseline_persistence"] = run_backtest(df, model_name="persist", **common)
        result["disclaimer"] = service.DISCLAIMER
        return result

    return _guard(run)
