import time

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import config
from app.data import providers
from app.data.providers import ProviderError, UnknownSymbol, get_asset, get_candles
from tests.conftest import synthetic_candles

HOUR_MS = 3_600_000


def recent_candles(n=1200, interval_ms=HOUR_MS, **kw):
    """Synthetic candles ending one full bar ago, so none is 'still forming'."""
    now_ms = int(time.time() * 1000)
    start = now_ms - (n + 1) * interval_ms
    return synthetic_candles(n, start_ms=start, step_ms=interval_ms, **kw)


class FakeResp:
    def __init__(self, payload):
        self._p = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._p


# ------------------------------------------------------------------- symbols
def test_symbol_resolution_and_injection_rejected():
    assert get_asset("btcusdt").symbol == "BTCUSDT"
    assert get_asset("XAUUSD").asset_class == "gold"
    assert get_asset("DOGEUSDT").asset_class == "crypto"       # any *USDT pair still works
    for bad in ["BTCUSDT&limit=1", "../etc", "BTC USDT", "", "EURXYZ"]:
        with pytest.raises(UnknownSymbol):
            get_asset(bad)


# ------------------------------------------------------------------ providers
def test_binance_parsing(monkeypatch):
    now = int(time.time() * 1000)
    raw = [[now - (3 - i) * HOUR_MS, "1", "2", "0.5", str(1 + i), "10", 0, 0, 0, 0, 0, 0] for i in range(3)]
    monkeypatch.setattr(providers._session, "get", lambda *a, **k: FakeResp(raw))
    df = get_candles("BTCUSDT", "1h", 3, drop_incomplete=False)
    assert list(df.columns) == providers.COLUMNS
    assert df["close"].tolist() == [1.0, 2.0, 3.0]


def test_binance_error_object_becomes_provider_error(monkeypatch):
    monkeypatch.setattr(providers._session, "get", lambda *a, **k: FakeResp({"code": -1121, "msg": "Invalid symbol."}))
    with pytest.raises(ProviderError):
        get_candles("BTCUSDT", "1h", 10)


def test_twelvedata_parsing_and_missing_key(monkeypatch):
    monkeypatch.setattr(config, "TWELVEDATA_API_KEY", "")
    with pytest.raises(ProviderError, match="TWELVEDATA_API_KEY"):
        get_candles("EURUSD", "1h", 10)

    monkeypatch.setattr(config, "TWELVEDATA_API_KEY", "k")
    payload = {"status": "ok", "values": [
        {"datetime": "2024-01-01 02:00:00", "open": "1.1", "high": "1.2", "low": "1.0", "close": "1.15"},
        {"datetime": "2024-01-01 01:00:00", "open": "1.0", "high": "1.1", "low": "0.9", "close": "1.05"},
    ]}
    monkeypatch.setattr(providers._session, "get", lambda *a, **k: FakeResp(payload))
    df = get_candles("EURUSD", "1h", 2)
    assert df["close"].tolist() == [1.05, 1.15]                  # sorted oldest first
    assert df["time"].iloc[0] == 1704070800000                   # 2024-01-01 01:00 UTC in ms
    assert (df["volume"] == 0).all()                             # FX feed has no volume


def test_twelvedata_error_payload(monkeypatch):
    monkeypatch.setattr(config, "TWELVEDATA_API_KEY", "k")
    monkeypatch.setattr(providers._session, "get",
                        lambda *a, **k: FakeResp({"code": 429, "message": "limit", "status": "error"}))
    with pytest.raises(ProviderError, match="limit"):
        get_candles("XAUUSD", "1h", 10)


# ---------------------------------------------------------- caching / fallback
def test_cache_prevents_repeat_fetches_and_stale_fallback(monkeypatch):
    calls = {"n": 0, "fail": False}

    def fake_fetch(asset, interval, limit):
        calls["n"] += 1
        if calls["fail"]:
            raise ProviderError("down")
        return recent_candles(50)

    monkeypatch.setattr(providers, "_fetch", fake_fetch)
    get_candles("BTCUSDT", "1m", 50)
    get_candles("BTCUSDT", "1m", 50)
    assert calls["n"] == 1                                       # second call served from cache

    # expire the entry (but keep it inside the stale window) and make the provider fail
    key = ("BTCUSDT", "1m", 50)
    providers._cache[key].stored_at -= 20
    calls["fail"] = True
    df = get_candles("BTCUSDT", "1m", 50)
    assert len(df) > 0 and calls["n"] == 2                       # retried, then served stale data

    providers._cache[key].stored_at -= 10_000                    # too old to serve
    with pytest.raises(ProviderError):
        get_candles("BTCUSDT", "1m", 50)


def test_incomplete_last_candle_is_dropped(monkeypatch):
    now = int(time.time() * 1000)
    df = pd.DataFrame({"time": [now - 2 * HOUR_MS, now - HOUR_MS - 1000, now - 1000],
                       "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0})
    monkeypatch.setattr(providers, "_fetch", lambda *a: df)
    assert len(get_candles("BTCUSDT", "1h", 3, drop_incomplete=True)) == 2
    providers.clear_cache()
    assert len(get_candles("BTCUSDT", "1h", 3, drop_incomplete=False)) == 3


# ------------------------------------------------------------------------ API
@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(providers, "_fetch", lambda asset, interval, limit: recent_candles(
        limit, config.INTERVALS[interval] * 1000, seed=11, phi=0.3))
    monkeypatch.setattr(config, "TWELVEDATA_API_KEY", "k")
    import main
    return TestClient(main.app)


def test_legacy_endpoints_still_work(client, tmp_path, monkeypatch):
    from app import legacy
    monkeypatch.setattr(legacy, "MODEL_PATH", str(tmp_path / "m.pkl"))
    assert client.get("/").json()["message"] == "AI Trading System Running"

    sig = client.get("/signals").json()
    assert [s["symbol"] for s in sig] == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    assert all({"price", "rsi", "signal"} <= s.keys() and s["price"] for s in sig)   # fields the UI reads

    pred = client.get("/predict/BTCUSDT").json()
    assert pred["prediction"] in ("UP", "DOWN")
    assert 50 <= pred["confidence"] <= 100                       # was probability-of-UP before

    chart = client.get("/chart/BTCUSDT").json()
    assert len(chart) == 50 and {"time", "open", "high", "low", "close"} <= chart[0].keys()


def test_legacy_signals_survive_provider_failure(client, monkeypatch):
    def boom(*a):
        raise ProviderError("exchange down")
    monkeypatch.setattr(providers, "_fetch", boom)
    sig = client.get("/signals")
    assert sig.status_code == 200                                # used to be a 500
    assert sig.json()[0]["signal"] == "N/A"
    assert client.get("/chart/BTCUSDT").status_code == 502


def test_v2_predict_all_asset_classes(client):
    for sym in ("BTCUSDT", "EURUSD", "XAUUSD"):
        r = client.get(f"/v2/predict/{sym}", params={"interval": "1h", "horizon": 4, "model": "logreg"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["horizon"] == "4h" and body["prediction"] in ("UP", "DOWN")
        assert 50 <= body["confidence"] <= 100
        assert "not financial advice" in body["disclaimer"]


def test_v2_validation_and_errors(client):
    assert client.get("/v2/predict/NOPE").status_code == 404
    assert client.get("/v2/predict/BTCUSDT", params={"interval": "2h"}).status_code == 422
    assert client.get("/v2/predict/BTCUSDT", params={"horizon": 0}).status_code == 422


def test_v2_signals_isolates_failures(client, monkeypatch):
    monkeypatch.setattr(config, "TWELVEDATA_API_KEY", "")        # forex/gold unavailable
    real = providers._fetch

    def fetch(asset, interval, limit):
        if asset.provider == "twelvedata":
            raise ProviderError("TWELVEDATA_API_KEY is not set")
        return real(asset, interval, limit)
    monkeypatch.setattr(providers, "_fetch", fetch)
    rows = client.get("/v2/signals", params={"model": "logreg"}).json()
    by = {r["symbol"]: r for r in rows}
    assert "prediction" in by["BTCUSDT"] and "error" in by["XAUUSD"]


def test_v2_backtest_endpoint(client):
    r = client.get("/v2/backtest/BTCUSDT", params={"model": "logreg", "step": 100})
    assert r.status_code == 200, r.text
    body = r.json()
    assert {"results", "baseline_persistence", "disclaimer"} <= body.keys()
    res = body["results"]
    assert res["target_hit_rate"] == 0.8 and isinstance(res["meets_target"], bool)
    assert len(res["calibration"]) == 4


def test_v2_symbols_lists_availability(client):
    rows = {r["symbol"]: r for r in client.get("/v2/symbols").json()}
    assert rows["XAUUSD"]["asset_class"] == "gold" and rows["BTCUSDT"]["available"] is True


def test_cors_is_not_wildcard(client):
    r = client.options("/", headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "GET"})
    assert "access-control-allow-origin" not in r.headers
    r = client.options("/", headers={"Origin": "http://localhost:3100", "Access-Control-Request-Method": "GET"})
    assert r.headers.get("access-control-allow-origin") == "http://localhost:3100"
