import pytest


def test_import_package():
    import cmf_week2
    from cmf_week2.config import SYMBOLS, TRAIN_START, VAL_END

    assert "btcusdt" in SYMBOLS
    assert TRAIN_START.year == 2025


def test_dense_liq_rolling():
    import polars as pl

    from cmf_week2.lib.features import _dense_liq_30s

    liq = pl.DataFrame({"sec": [100, 102], "liq_signed": [10.0, 5.0], "liq_notional": [10.0, 5.0], "liq_count": [1, 1]})
    out = _dense_liq_30s(liq, "bb", 100, 102)
    row101 = out.filter(pl.col("sec") == 101)
    assert row101["bb_l_net_30s"][0] == 10.0


def test_heuristic_threshold_from_train_quantile():
    import numpy as np
    import polars as pl

    from cmf_week2.lib.filter import heuristic_filter_calibrated

    train = pl.DataFrame(
        {
            "bb_l_net_30s": [10.0, 20.0, 30.0, 40.0],
            "pnl_30s_bps": [1.0, 1.0, 1.0, 1.0],
            "w": [1e9] * 4,
            "timestamp": [1, 2, 3, 4],
        }
    )
    test = pl.DataFrame(
        {
            "bb_l_net_30s": [50.0, 5.0],
            "pnl_30s_bps": [0.0, 0.0],
            "w": [1e9, 1e9],
            "timestamp": [10, 11],
        }
    )
    expected_thr = float(np.nanquantile(train["bb_l_net_30s"].to_numpy(), 0.85))
    r = heuristic_filter_calibrated(train, test, "pnl_30s_bps", quantiles=(0.85,))
    assert r.threshold == expected_thr
    assert r.f[0] == int(abs(50.0) >= expected_thr)
    assert r.f[1] == int(abs(5.0) >= expected_thr)


def test_run_baseline_smoke(tmp_path, monkeypatch):
    pytest.importorskip("polars")
    from cmf_week2.config import has_raw_data

    if not has_raw_data(("btcusdt",)):
        pytest.skip("no parquet — set LIQUIDATION_DATA_ROOT for integration smoke")

    from cmf_week2.cli import run_baseline

    monkeypatch.setenv("LIQUIDATION_DATA_ROOT", str(__import__("cmf_week2.config", fromlist=["DATA"]).DATA))
    rc = run_baseline.main(["--symbol", "btcusdt", "--max-trades", "5000"])
    assert rc == 0
