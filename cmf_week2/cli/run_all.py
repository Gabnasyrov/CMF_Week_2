"""Run full evaluation pipeline."""

from __future__ import annotations

import sys

from cmf_week2.cli import export_lgbm_pnl, plot_lgbm_auc, run_baseline


def main(argv: list[str] | None = None) -> int:
    args = list(argv or sys.argv[1:])
    rc = run_baseline.main(["--all-symbols"] + [a for a in args if a.startswith("--max")])
    if rc != 0:
        return rc
    rc = export_lgbm_pnl.main([])
    if rc != 0:
        return rc
    if "--skip-lgbm-roc" not in args:
        return plot_lgbm_auc.main([])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
