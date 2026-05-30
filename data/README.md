# Parquet data (not in git)

Place 6-month archive locally:

```
binance_trades/perp_{btcusdt,ethusdt}.parquet
binance_booktickers/perp_*.parquet
binance_liquidations/perp_*.parquet
bybit_liquidations/{btcusdt,ethusdt}.parquet
```

`timestamp`: int64, microseconds UTC. Bybit liquidations: apply **+200 ms** before joining Binance.

Set `LIQUIDATION_DATA_ROOT` to this folder. See [README.md](../README.md) for schema.
