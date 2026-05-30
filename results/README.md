# Bundled results

| File | Description |
|------|-------------|
| `baseline_metrics_full.csv` | Full test metrics from `tz_assignment` LGBM pipeline (Binance trades, both symbols) |
| `baseline_metrics.csv` | Polars baseline sample run (`run_baseline.py`) |
| `REPORT.md` | One-page summary for weekly report |

Regenerate Polars sample:

```bash
python scripts/run_baseline.py --symbol btcusdt --max-trades 500000
```
