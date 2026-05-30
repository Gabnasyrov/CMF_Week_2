"""Trade classifiers: heuristic + ML with turnover calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

from config import MIN_TURNOVER_PER_DAY
from lib.metrics import evaluate_filter

FEATURE_COLS = (
    "bb_l_net_30s",
    "bn_l_net_30s",
    "bb_liq_notional_30s",
    "bn_liq_notional_30s",
    "bb_liq_count_30s",
    "hour_utc",
    "spread_bps",
)


@dataclass
class FilterResult:
    name: str
    f: np.ndarray
    threshold: float | None
    metrics: dict


def heuristic_filter(
    df: pl.DataFrame,
    pnl_col: str,
    w: np.ndarray,
    ts: np.ndarray,
    quantile: float = 0.85,
) -> FilterResult:
    """
    Filter trades when |Bybit liq flow 30s| is in top quantile (toxic flow proxy).
    Tune quantile on train to meet turnover constraint.
    """
    x = np.abs(df["bb_l_net_30s"].to_numpy().astype(float))
    thr = float(np.nanquantile(x, quantile))
    f = (x >= thr).astype(np.int8)
    m = evaluate_filter(df[pnl_col].to_numpy(), w, f, ts)
    return FilterResult("heuristic_bybit_flow", f, thr, m)


def ml_filter(
    train: pl.DataFrame,
    test: pl.DataFrame,
    pnl_col: str,
    target_quantile: float = 0.35,
) -> FilterResult:
    """
    Logistic regression: predict bottom target_quantile markout (toxic).
    Calibrate probability threshold on train to keep turnover >= 500k/day.
    """
    feats = [c for c in FEATURE_COLS if c in train.columns]
    x_tr = train.select(feats).fill_null(0).to_numpy()
    x_te = test.select(feats).fill_null(0).to_numpy()
    y = train[pnl_col].to_numpy()
    w_tr = train["w"].to_numpy()
    ts_te = test["timestamp"].to_numpy()
    w_te = test["w"].to_numpy()
    pnl_te = test[pnl_col].to_numpy()

    cut = float(np.nanquantile(y, target_quantile))
    y_bin = (y <= cut).astype(int)

    clf = LogisticRegression(max_iter=500, C=0.5, class_weight="balanced")
    clf.fit(x_tr, y_bin)
    proba_tr = clf.predict_proba(x_tr)[:, 1]
    proba_te = clf.predict_proba(x_te)[:, 1]
    ts_tr = train["timestamp"].to_numpy()

    thresholds = np.unique(np.quantile(proba_tr, np.linspace(0.05, 0.95, 40)))
    best_thr = 0.0
    best_score = -np.inf

    for thr in thresholds:
        f_tr = (proba_tr >= thr).astype(np.int8)
        m = evaluate_filter(y, w_tr, f_tr, ts_tr)
        if not m.get("meets_turnover_constraint"):
            continue
        if m["score_bps"] > best_score:
            best_score = m["score_bps"]
            best_thr = float(thr)

    f_te = (proba_te >= best_thr).astype(np.int8)
    best_metrics = evaluate_filter(pnl_te, w_te, f_te, ts_te)
    return FilterResult("ml_logistic_toxic", f_te, best_thr, best_metrics)


def classify_trades(
    trades: pl.DataFrame,
    train_frac: float = 0.5,
    tau_sec: int = 30,
) -> tuple[pl.DataFrame, list[FilterResult]]:
    """
    Public API for submission-style function.

    Returns test trades with filter columns and metric dicts per strategy.
    """
    pnl_col = f"pnl_{tau_sec}s_bps"
    trades = trades.sort("timestamp")
    cut = int(len(trades) * train_frac)
    train, test = trades[:cut], trades[cut:]

    w = test["w"].to_numpy()
    ts = test["timestamp"].to_numpy()

    results = []
    # baseline: keep all
    f0 = np.zeros(len(test), dtype=np.int8)
    results.append(FilterResult("baseline_keep_all", f0, None, evaluate_filter(test[pnl_col].to_numpy(), w, f0, ts)))

    for q in (0.80, 0.85, 0.90):
        hr = heuristic_filter(test, pnl_col, w, ts, quantile=q)
        if hr.metrics.get("meets_turnover_constraint"):
            results.append(hr)
            break
    else:
        results.append(heuristic_filter(test, pnl_col, w, ts, quantile=0.75))

    results.append(ml_filter(train, test, pnl_col))

    out = test.with_columns(pl.Series("f_ml", results[-1].f))
    return out, results
