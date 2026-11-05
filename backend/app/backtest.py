"""Walk-forward backtesting with costs, confidence intervals and calibration.

Honest evaluation rules enforced here:
* A prediction made at bar ``t`` only uses a model trained on rows whose label
  was already known at ``t`` (row s is usable only if s + horizon <= t).
* Costs (fee + slippage, both sides) are subtracted from every trade.
* Hit rates come with a 95% Wilson confidence interval, and the "meets target"
  flag requires the LOWER bound to clear the target on enough trades, so a
  lucky small sample can't pass.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config
from .features import WARMUP, build_features, forward_log_return
from .models import fit_model, prob_up


# ----------------------------------------------------------------- statistics
def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


# ---------------------------------------------------------------- walk-forward
def walk_forward(
    df: pd.DataFrame,
    horizon: int,
    model_name: str,
    min_train: int = 300,
    step: int = 50,
    max_train: int | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """Out-of-sample predictions for every bar after the first ``min_train``.

    Returns (predictions, folds). ``predictions`` has columns
    pos, time, prob_up, fwd (realised log return over the horizon).
    ``folds`` documents each retraining (used by the leakage tests).
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    feat = build_features(df)
    fwd = forward_log_return(df, horizon)
    valid = feat.notna().all(axis=1).to_numpy()
    n = len(df)
    if not valid.any():
        raise ValueError(f"need at least {WARMUP + 1} candles, got {n}")
    first = int(np.argmax(valid))

    start = first + min_train + horizon
    last_pred = n - 1 - horizon  # last bar whose outcome is already known
    if start > last_pred:
        raise ValueError(
            f"not enough data: have {n} candles, need more than {start + horizon + 1} "
            f"for min_train={min_train}, horizon={horizon}"
        )

    fwd_np = fwd.to_numpy()
    rows, folds = [], []
    for i in range(start, last_pred + 1, step):
        train_end = i - horizon              # inclusive; label of this row is known at bar i
        train_start = first if max_train is None else max(first, train_end - max_train + 1)
        idx = np.arange(train_start, train_end + 1)
        idx = idx[valid[idx] & ~np.isnan(fwd_np[idx])]
        model = fit_model(model_name, feat.iloc[idx], (fwd_np[idx] > 0).astype(int))

        test = np.arange(i, min(i + step, last_pred + 1))
        test = test[valid[test]]
        p = prob_up(model, feat.iloc[test])
        rows.append(pd.DataFrame({
            "pos": test, "time": df["time"].to_numpy()[test], "prob_up": p, "fwd": fwd_np[test],
        }))
        folds.append({"test_start": i, "train_end": int(train_end), "n_train": int(len(idx))})
    return pd.concat(rows, ignore_index=True), folds


# ------------------------------------------------------------------ evaluation
@dataclass
class EvalConfig:
    horizon: int
    interval: str
    threshold: float = 0.55        # minimum confidence to take a trade
    fee_bps: float = 10.0          # one-way
    slippage_bps: float = 1.0      # one-way
    allow_short: bool = True
    trading_days_per_year: int = 365
    target_hit_rate: float = 0.80
    min_trades_for_target: int = 200


def evaluate(preds: pd.DataFrame, cfg: EvalConfig) -> dict:
    """Turn out-of-sample predictions into trading and accuracy statistics."""
    p = preds["prob_up"].to_numpy()
    fwd = preds["fwd"].to_numpy()
    n_pred = len(preds)
    direction = np.where(p >= 0.5, 1, -1)
    conf = np.maximum(p, 1 - p)
    correct = (direction > 0) == (fwd > 0)

    hit_all = int(correct.sum())
    lo_all, hi_all = wilson_interval(hit_all, n_pred)

    take = conf >= cfg.threshold
    if not cfg.allow_short:
        take &= direction > 0
    n_trades = int(take.sum())
    hit_tr = int(correct[take].sum())
    lo_tr, hi_tr = wilson_interval(hit_tr, n_trades)

    cost = 2 * (cfg.fee_bps + cfg.slippage_bps) / 1e4  # round trip, fraction of notional
    net = direction * np.expm1(fwd) - cost
    net_trades = net[take]

    wins, losses = net_trades[net_trades > 0].sum(), -net_trades[net_trades < 0].sum()
    profit_factor = float(wins / losses) if losses > 0 else (float("inf") if wins > 0 else 0.0)

    # Equity curve: capital is split across the `horizon` overlapping slots, and
    # each trade's P&L lands on its exit bar.
    sharpe, max_dd, final_equity = 0.0, 0.0, 1.0
    if n_trades:
        pos = preds["pos"].to_numpy()[take]
        exit_pos = pos + cfg.horizon
        span = int(exit_pos.max() - pos.min()) + 1
        pnl = np.zeros(span)
        np.add.at(pnl, exit_pos - pos.min(), net_trades / cfg.horizon)
        equity = np.cumprod(1 + pnl)
        final_equity = float(equity[-1])
        max_dd = float((equity / np.maximum.accumulate(equity) - 1).min())
        bars_per_year = cfg.trading_days_per_year * 86400 / config.INTERVALS[cfg.interval]
        if pnl.std() > 0:
            sharpe = float(pnl.mean() / pnl.std() * math.sqrt(bars_per_year))

    # Calibration: when the model says X% confident, how often is it right?
    edges = [0.5, 0.55, 0.6, 0.7, 1.0001]
    calibration = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (conf >= a) & (conf < b)
        calibration.append({
            "confidence_range": f"{a:.2f}-{min(b, 1.0):.2f}",
            "n": int(m.sum()),
            "mean_confidence": round(float(conf[m].mean()), 4) if m.any() else None,
            "hit_rate": round(float(correct[m].mean()), 4) if m.any() else None,
        })

    meets_target = bool(
        n_trades >= cfg.min_trades_for_target and lo_tr >= cfg.target_hit_rate
    )

    flags = []
    if n_trades < 100:
        flags.append(f"Only {n_trades} trades: too few to draw conclusions.")
    if n_pred >= 100 and hit_all / n_pred > 0.65:
        flags.append(
            "Accuracy above 65% on liquid markets is very unusual. Check for "
            "look-ahead leakage before trusting it."
        )
    if n_trades >= 30 and lo_tr <= 0.5:
        flags.append("Trade hit-rate confidence interval includes 50%: no proven edge over a coin flip.")
    if n_trades and float(net_trades.mean()) <= 0:
        flags.append("Average trade loses money after costs.")

    return {
        "n_predictions": n_pred,
        "n_trades": n_trades,
        "coverage": round(n_trades / n_pred, 4) if n_pred else 0.0,
        "accuracy_all_predictions": round(hit_all / n_pred, 4) if n_pred else None,
        "accuracy_all_ci95": [round(lo_all, 4), round(hi_all, 4)],
        "accuracy_trades": round(hit_tr / n_trades, 4) if n_trades else None,
        "accuracy_trades_ci95": [round(lo_tr, 4), round(hi_tr, 4)],
        "baseline_always_up": round(float((fwd > 0).mean()), 4) if n_pred else None,
        "avg_net_return_per_trade_bps": round(float(net_trades.mean()) * 1e4, 2) if n_trades else None,
        "profit_factor": None if math.isinf(profit_factor) else round(profit_factor, 3),
        "sharpe_annualised": round(sharpe, 3),
        "max_drawdown": round(max_dd, 4),
        "final_equity_multiple": round(final_equity, 4),
        "assumed_cost_bps_per_side": cfg.fee_bps + cfg.slippage_bps,
        "calibration": calibration,
        "target_hit_rate": cfg.target_hit_rate,
        "meets_target": meets_target,
        "warnings": flags,
    }


def run_backtest(
    df: pd.DataFrame,
    *,
    interval: str,
    horizon: int,
    model_name: str,
    asset: config.Asset,
    threshold: float = 0.55,
    min_train: int = 300,
    step: int = 50,
    max_train: int | None = None,
    slippage_bps: float | None = None,
    allow_short: bool = True,
    target_hit_rate: float = 0.80,
) -> dict:
    cfg = EvalConfig(
        horizon=horizon, interval=interval, threshold=threshold,
        fee_bps=asset.fee_bps,
        slippage_bps=config.DEFAULT_SLIPPAGE_BPS if slippage_bps is None else slippage_bps,
        allow_short=allow_short,
        trading_days_per_year=asset.periods_per_year_days,
        target_hit_rate=target_hit_rate,
    )
    preds, folds = walk_forward(df, horizon, model_name, min_train, step, max_train)
    result = evaluate(preds, cfg)
    result["folds"] = len(folds)
    result["period"] = {
        "from_ms": int(preds["time"].iloc[0]), "to_ms": int(preds["time"].iloc[-1]),
    }
    return result
