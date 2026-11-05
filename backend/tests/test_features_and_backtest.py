import numpy as np
import pandas as pd
import pytest

from app import config
from app.backtest import EvalConfig, evaluate, run_backtest, walk_forward, wilson_interval
from app.features import FEATURE_COLUMNS, WARMUP, build_features, rsi
from tests.conftest import synthetic_candles


# ------------------------------------------------------------------- features
def test_features_are_causal():
    """Features at row t must be identical whether or not future rows exist."""
    df = synthetic_candles(400, seed=1)
    full = build_features(df)
    for cut in (120, 250, 399):
        part = build_features(df.iloc[: cut + 1].copy())
        pd.testing.assert_series_equal(
            full.iloc[cut], part.iloc[cut], check_names=False, rtol=1e-9, atol=1e-12
        )


def test_feature_columns_defined_after_warmup():
    f = build_features(synthetic_candles(300, seed=2))
    assert list(f.columns) == FEATURE_COLUMNS
    assert f.iloc[WARMUP:].notna().all().all()


def test_rsi_bounds_and_flat_market():
    close = pd.Series(np.linspace(100, 200, 100))
    assert rsi(close).dropna().between(0, 100).all()
    assert rsi(close).iloc[-1] == pytest.approx(100)
    assert rsi(pd.Series([50.0] * 60)).iloc[-1] == 50.0


def test_features_handle_zero_volume():
    f = build_features(synthetic_candles(200, seed=3, volume=False))
    assert (f["vol_z"] == 0).all()


# ---------------------------------------------------------------- statistics
def test_wilson_interval():
    lo, hi = wilson_interval(80, 100)
    assert lo == pytest.approx(0.7112, abs=1e-3) and hi == pytest.approx(0.8666, abs=1e-3)
    assert wilson_interval(0, 0) == (0.0, 1.0)


# ------------------------------------------------------------------ no leakage
def test_walk_forward_folds_respect_label_availability():
    df = synthetic_candles(900, seed=4)
    for horizon in (1, 4):
        _, folds = walk_forward(df, horizon, "logreg", min_train=200, step=40)
        assert folds
        for fold in folds:
            # the newest training label resolves at train_end + horizon,
            # which must not be later than the first bar we predict
            assert fold["train_end"] + horizon <= fold["test_start"]


def test_predictions_do_not_depend_on_future_prices():
    df = synthetic_candles(900, seed=5)
    base, _ = walk_forward(df, 2, "hgb", min_train=200, step=40)
    cutoff = 600
    tampered = df.copy()
    rng = np.random.default_rng(99)
    tampered.loc[cutoff + 1:, ["open", "high", "low", "close"]] *= rng.uniform(0.5, 2.0, size=(len(df) - cutoff - 1, 1))
    changed, _ = walk_forward(tampered, 2, "hgb", min_train=200, step=40)
    a = base[base["pos"] <= cutoff].set_index("pos")["prob_up"]
    b = changed[changed["pos"] <= cutoff].set_index("pos")["prob_up"]
    pd.testing.assert_series_equal(a, b, check_exact=False, rtol=1e-9)


# --------------------------------------------------------------- evaluation math
def test_evaluate_hand_computed():
    preds = pd.DataFrame({
        "pos": [0, 1, 2, 3],
        "time": [0, 1, 2, 3],
        "prob_up": [0.9, 0.2, 0.8, 0.51],           # UP, DOWN, UP, UP(weak)
        "fwd": [0.01, -0.02, -0.01, 0.01],          # right, right, wrong, right
    })
    cfg = EvalConfig(horizon=1, interval="1h", threshold=0.6, fee_bps=10, slippage_bps=0,
                     min_trades_for_target=1)
    r = evaluate(preds, cfg)
    assert r["n_predictions"] == 4
    assert r["accuracy_all_predictions"] == 0.75
    assert r["n_trades"] == 3                       # the 0.51 call is below the 0.6 threshold
    assert r["accuracy_trades"] == pytest.approx(2 / 3, abs=1e-4)
    cost = 0.0020
    expected = np.mean([np.expm1(0.01) - cost, np.expm1(-0.02) * -1 - cost, np.expm1(-0.01) - cost])
    assert r["avg_net_return_per_trade_bps"] == pytest.approx(expected * 1e4, abs=0.01)


def test_no_short_mode_skips_down_calls():
    preds = pd.DataFrame({"pos": [0, 1], "time": [0, 1], "prob_up": [0.9, 0.1], "fwd": [0.01, -0.01]})
    cfg = EvalConfig(horizon=1, interval="1h", threshold=0.55, allow_short=False)
    assert evaluate(preds, cfg)["n_trades"] == 1


# ------------------------------------------------------------ honesty of results
def test_random_walk_shows_no_edge_and_cannot_pass_target():
    df = synthetic_candles(1500, seed=6, phi=0.0)
    asset = config.ASSETS["BTCUSDT"]
    res = run_backtest(df, interval="1h", horizon=1, model_name="logreg", asset=asset, min_train=300, step=50)
    assert 0.44 < res["accuracy_all_predictions"] < 0.56
    assert res["meets_target"] is False
    assert res["avg_net_return_per_trade_bps"] is None or res["avg_net_return_per_trade_bps"] < 5


def test_model_finds_a_real_signal_when_one_exists():
    """Autocorrelated returns are genuinely predictable; the pipeline must find it."""
    df = synthetic_candles(2000, seed=7, phi=0.5)
    asset = config.ASSETS["EURUSD"]
    res = run_backtest(df, interval="1h", horizon=1, model_name="logreg", asset=asset, min_train=300, step=100)
    assert res["accuracy_all_predictions"] > 0.57
    assert res["accuracy_all_ci95"][0] > 0.5


def test_not_enough_data_raises():
    with pytest.raises(ValueError):
        walk_forward(synthetic_candles(200), 1, "logreg", min_train=300)
