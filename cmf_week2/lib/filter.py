"""Trade classifiers — threshold tuning on train only."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

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


@dataclass
class FilterFit:
    heuristic_thr: float
    ml_clf: LogisticRegression
    ml_thr: float
    ml_feats: tuple[str, ...]


def _fit_heuristic_threshold(
    train: pl.DataFrame,
    pnl_col: str,
    quantiles: tuple[float, ...] = (0.75, 0.80, 0.85, 0.90),
) -> float:
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
    return best_thr


def _fit_ml_filter(
    train: pl.DataFrame,
    pnl_col: str,
    target_quantile: float = 0.35,
) -> tuple[LogisticRegression, float, tuple[str, ...]]:
    feats = tuple(c for c in FEATURE_COLS if c in train.columns)
    x_tr = train.select(list(feats)).fill_null(0).to_numpy()
    y = train[pnl_col].to_numpy()
    w_tr = train["w"].to_numpy()
    ts_tr = train["timestamp"].to_numpy()
    cut = float(np.nanquantile(y, target_quantile))
    y_bin = (y <= cut).astype(int)
    clf = LogisticRegression(max_iter=500, C=0.5, class_weight="balanced")
    clf.fit(x_tr, y_bin)
    proba_tr = clf.predict_proba(x_tr)[:, 1]
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
    return clf, best_thr, feats


def fit_filters(train: pl.DataFrame, tau_sec: int) -> FilterFit:
    pnl_col = f"pnl_{tau_sec}s_bps"
    h_thr = _fit_heuristic_threshold(train, pnl_col)
    clf, ml_thr, feats = _fit_ml_filter(train, pnl_col)
    return FilterFit(heuristic_thr=h_thr, ml_clf=clf, ml_thr=ml_thr, ml_feats=feats)


def _apply_heuristic(df: pl.DataFrame, thr: float) -> np.ndarray:
    x = np.abs(df["bb_l_net_30s"].to_numpy().astype(float))
    return (x >= thr).astype(np.int8)


def _apply_ml_filter(df: pl.DataFrame, fit: FilterFit) -> np.ndarray:
    x = df.select(list(fit.ml_feats)).fill_null(0).to_numpy()
    proba = fit.ml_clf.predict_proba(x)[:, 1]
    return (proba >= fit.ml_thr).astype(np.int8)


def apply_filter(df: pl.DataFrame, fit: FilterFit, strategy: str) -> np.ndarray:
    if strategy == "baseline_keep_all":
        return np.zeros(len(df), dtype=np.int8)
    if strategy == "heuristic_bybit_flow":
        return _apply_heuristic(df, fit.heuristic_thr)
    if strategy == "ml_logistic_toxic":
        return _apply_ml_filter(df, fit)
    raise ValueError(strategy)


def classify_trades(trades: pl.DataFrame, tau_sec: int = 30) -> tuple[pl.DataFrame, list[FilterResult]]:
    """Fit on train (Dec–Jan), evaluate on Feb validation."""
    pnl_col = f"pnl_{tau_sec}s_bps"
    trades = trades.sort("timestamp")
    train, test = official_train_test(trades)
    if len(test) == 0:
        raise ValueError("Empty test window — check VAL_START/VAL_END and loaded data range")

    fit = fit_filters(train, tau_sec)
    results: list[FilterResult] = []
    for name in ("baseline_keep_all", "heuristic_bybit_flow", "ml_logistic_toxic"):
        f = apply_filter(test, fit, name)
        thr = None
        if name == "heuristic_bybit_flow":
            thr = fit.heuristic_thr
        elif name == "ml_logistic_toxic":
            thr = fit.ml_thr
        results.append(
            FilterResult(
                name,
                f,
                thr,
                evaluate_filter(test[pnl_col].to_numpy(), test["w"].to_numpy(), f, test["timestamp"].to_numpy()),
            )
        )
    return test, results
