"""Import LGBM PnL from liquidation_task or bundled pnl_report."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cmf_week2.config import RESULTS, liquidation_task_root


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=None, help="pnl_report.csv path")
    args = p.parse_args(argv)

    src = args.source
    if src is None:
        root = liquidation_task_root()
        if root is not None:
            candidate = root / "tz_assignment" / "results" / "strategies" / "pnl_report.csv"
            if candidate.is_file():
                src = candidate
    if src is None or not src.is_file():
        bundled = RESULTS / "baseline_metrics_full.csv"
        print(f"No external pnl_report — keeping existing {bundled}", flush=True)
        return 0

    df = pd.read_csv(src)
    df = df[df["venue"] == "binance"].copy()
    df["source"] = "liquidation_task_pnl_report"
    df.to_csv(RESULTS / "baseline_metrics_full.csv", index=False)

    note = RESULTS / "baseline_metrics_full_README.txt"
    note.write_text(
        "baseline_metrics_full.csv imported from liquidation_task pnl_report (train_50_test_50 split).\n"
        "n_days differs by symbol because test span after markout drop differs (~10d BTC, ~46d ETH).\n"
        "Polars baseline with official calendar split: results/baseline_metrics.csv\n",
        encoding="utf-8",
    )
    print(f"Wrote {RESULTS / 'baseline_metrics_full.csv'} ({len(df)} rows) from {src}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
