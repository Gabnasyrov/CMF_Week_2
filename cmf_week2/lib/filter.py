"""Trade classifiers — threshold tuning on train only."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

from cmf_week2.config import MIN_TURNOVER_PER_DAY
from cmf_week2.lib.metrics import evaluate_filter
from cmf_week2.lib.split import official_train_test

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


def heuristic_filter_calibrated(
    train: pl.DataFrame,
    test: pl.DataFrame,
    pnl_col: str,
    quantiles: tuple[float, ...] = (0.75, 0.80, 0.85, 0.90),
) -> FilterResult:
    """Pick |Bybit flow| quantile threshold on train; apply to test."""
    x_tr = np.abs(train["bb_l_net_30s"].to_numpy().astype(float))
    y_tr = train[pnl_col].to_numpy()
    w_tr = train["w"].to_numpy()
    ts_tr = train["timestamp"].to_numpy()

    best_thr = float(np.nanquantile(x_tr, 0.85))
    best_score = -np.inf
    for q in quantiles:
        thr = float(np.nanquantile(x_tr, q))
        f_tr = (x_tr >= thr).astype(np.int8)
        m = evaluate_filter(y_tr, w_tr, f_tr, ts_tr)
        if not m.get("meets_turnover_constraint"):
            continue
        if m["score_bps"] > best_score:
            best_score = m["score_bps"]
            best_thr = thr

    x_te = np.abs(test["bb_l_net_30s"].to_numpy().astype(float))
    f_te = (x_te >= best_thr).astype(np.int8)
    m_te = evaluate_filter(
        test[pnl_col].to_numpy(),
        test["w"].to_numpy(),
        f_te,
        test["timestamp"].to_numpy(),
    )
    return FilterResult("heuristic_bybit_flow", f_te, best_thr, m_te)


def ml_filter(train: pl.DataFrame, test: pl.DataFrame, pnl_col: str, target_quantile: float = 0.35) -> FilterResult:
    """Logistic regression; probability threshold tuned on train only."""
    feats = [c for c in FEATURE_COLS if c in train.columns]
    x_tr = train.select(feats).fill_null(0).to_numpy()
    x_te = test.select(feats).fill_null(0).to_numpy()
    y = train[pnl_col].to_numpy()
    w_tr = train["w"].to_numpy()
    ts_tr = train["timestamp"].to_numpy()
    ts_te = test["timestamp"].to_numpy()
    w_te = test["w"].to_numpy()
    pnl_te = test[pnl_col].to_numpy()

    cut = float(np.nanquantile(y, target_quantile))
    y_bin = (y <= cut).astype(int)

    clf = LogisticRegression(max_iter=500, C=0.5, class_weight="balanced")
    clf.fit(x_tr, y_bin)
    proba_tr = clf.predict_proba(x_tr)[:, 1]
    proba_te = clf.predict_proba(x_te)[:, 1]

    thresholds = np.unique(np.quantile(proba_tr, np.linspace(0.05, 0.95, 40)))
    best_thr = 1.0
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
    return FilterResult("ml_logistic_toxic", f_te, best_thr, evaluate_filter(pnl_te, w_te, f_te, ts_te))


def classify_trades(trades: pl.DataFrame, tau_sec: int = 30) -> tuple[pl.DataFrame, list[FilterResult]]:
    """Official calendar split: fit thresholds on train, evaluate on Feb validation."""
    pnl_col = f"pnl_{tau_sec}s_bps"
    trades = trades.sort("timestamp")
    train, test = official_train_test(trades)

    if len(test) == 0:
        raise ValueError("Empty test window — check VAL_START/VAL_END and loaded data range")

    results: list[FilterResult] = []
    f0 = np.zeros(len(test), dtype=np.int8)
    results.append(
        FilterResult(
            "baseline_keep_all",
            f0,
            None,
            evaluate_filter(test[pnl_col].to_numpy(), test["w"].to_numpy(), f0, test["timestamp"].to_numpy()),
        )
    )
    results.append(heuristic_filter_calibrated(train, test, pnl_col))
    results.append(ml_filter(train, test, pnl_col))

    out = test.with_columns(pl.Series("f_ml", results[-1].f))
    return out, results
