#!/usr/bin/env python3
"""Train LGBM no-K @ 30s on panel and save ROC / AUC figures for reviewer."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import RocCurveDisplay, auc, roc_curve

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "research" / "strategies_bundle"
sys.path.insert(0, str(BUNDLE))

from lib.config import FEATURES_NO_KALMAN  # noqa: E402
from lib.feature_builder import add_hawkes_features  # noqa: E402
from lib.io_config import load_symbol_config  # noqa: E402
from lib.models import LGBMDirectionModel  # noqa: E402
from lib.pipeline import chronological_split, fit_normalizers, prepare_panel, transform_panel  # noqa: E402
from lib.targets import lgbm_train_mask  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"
FIG = OUT / "figures"
HORIZON = 30


def _binary(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = np.isfinite(y) & (y != 0)
    return m, (y[m] > 0).astype(int)


def run_symbol(sym: str, max_days: int | None = 90) -> dict:
    cfg = load_symbol_config(sym)
    panel = prepare_panel(sym, max_days=max_days, skip_hawkes=True)
    tr, te = chronological_split(panel, cfg.get("train_frac", 0.4))
    train_end_us = int(tr["ts_us"].iloc[-1])
    panel = add_hawkes_features(panel, sym, train_end_us)
    tr, te = chronological_split(panel, cfg.get("train_frac", 0.4))
    mtr = np.zeros(len(panel), dtype=bool)
    mtr[: len(tr)] = True

    _, norm_nk = fit_normalizers(tr)
    _, zn_te = transform_panel(te, norm_nk, norm_nk)
    zn_tr = transform_panel(tr, norm_nk, norm_nk)[1]
    zcols = norm_nk.z_columns()

    ycol = f"y_mu_sign_{HORIZON}s"
    m_fit = lgbm_train_mask(panel, mtr, ycol, HORIZON)
    model = LGBMDirectionModel(**cfg["lgbm"])
    model.fit(zn_tr.loc[m_fit[mtr], zcols], panel.loc[m_fit, ycol].values, zn_te[zcols], te[ycol].values)

    proba = model.model.predict_proba(zn_te[zcols])[:, 1]
    m_te, y_bin = _binary(te[ycol].values)
    fpr, tpr, _ = roc_curve(y_bin, proba[m_te])
    roc_auc = float(auc(fpr, tpr))

    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=f"LGBM no-K @ {HORIZON}s").plot(ax=ax)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_title(f"{sym.upper()} — direction classifier (test split)")
    fig.tight_layout()
    fig_path = FIG / f"lgbm_roc_{sym}_{HORIZON}s.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)

    pred = model.predict_sign(zn_te[zcols])
    hit = float((np.sign(te[ycol].values[m_te]) == pred[m_te]).mean())

    return {
        "symbol": sym,
        "horizon_sec": HORIZON,
        "features": FEATURES_NO_KALMAN,
        "n_features": len(FEATURES_NO_KALMAN),
        "lgbm_params": model.params,
        "train_rows": int(m_fit.sum()),
        "test_rows": int(m_te.sum()),
        "auc_test": roc_auc,
        "hit_rate_test": hit,
        "roc_figure": str(fig_path.name),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    rows = []
    for sym in ("btcusdt", "ethusdt"):
        print(f"Plotting ROC for {sym}...", flush=True)
        rows.append(run_symbol(sym))
    meta = {
        "model": "lgbm_nk_30s",
        "target": f"y_mu_sign_{HORIZON}s = sign(kalman_mu[t+{HORIZON}s] - kalman_mu[t])",
        "normalization": "train-only z-score, clip ±8",
        "symbols": rows,
    }
    path = OUT / "lgbm_direction_metrics.json"
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {path} and ROC figures in {FIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
