"""
Unit tests for binance_vision/build_master_features.py
"""
import polars as pl
import pytest

from binance_vision.build_master_features import build_master_feature_table


def test_build_master_feature_table_basic():
    # Aligned minute timestamps:
    # 22:14:00.000 (1700000040000) and 22:15:00.000 (1700000100000)
    t1 = 1700000040000
    t2 = 1700000100000

    # Mock Futures 1m Klines
    df_um = pl.DataFrame({
        "close_time": [t1, t2],
        "open": [100.0, 102.0],
        "high": [105.0, 106.0],
        "low": [99.0, 101.0],
        "close": [103.0, 105.0],
        "buy_vol": [10.0, 15.0],
        "sell_vol": [5.0, 8.0],
        "count": [100, 120],
    })

    # Mock Spot 1m Klines
    df_spot = pl.DataFrame({
        "close_time": [t1, t2],
        "close": [100.0, 102.0],
        "buy_vol": [8.0, 12.0],
        "sell_vol": [4.0, 6.0],
        "count": [80, 90],
    })

    # Mock aggTrades in interval (22:13:00, 22:14:00] -> timestamp label="right" is 22:14:00 (t1)
    df_agg = pl.DataFrame({
        "agg_trade_id": [1, 2],
        "price": [101.0, 102.0],
        "quantity": [1.0, 2.0],
        "first_trade_id": [10, 20],
        "last_trade_id": [10, 22],
        "transact_time": [1699999990000, 1700000010000],  # 22:13:10 and 22:13:30
        "is_buyer_maker": [False, True],
        "is_best_match": [True, True],
    })

    # Mock Metrics
    df_met = pl.DataFrame({
        "create_time": [t1],
        "sum_open_interest": [1000.0],
        "sum_open_interest_value": [500000.0],
        "count_toptrader_long_short_ratio": [1.5],
        "sum_toptrader_long_short_ratio": [1.6],
        "count_long_short_ratio": [1.2],
        "sum_taker_long_short_vol_ratio": [1.1],
    })

    master = build_master_feature_table(df_um, df_spot, df_aggtrades=df_agg, df_metrics=df_met)

    assert len(master) == 2
    assert "basis" in master.columns
    assert "basis_rate" in master.columns
    assert master["basis"][0] == pytest.approx(3.0)
    assert master["basis_rate"][0] == pytest.approx(0.03)

    # Verify aggTrades features joined at close_time t1
    assert "vwap_disparity" in master.columns
    assert master["vwap_disparity"][0] is not None
    assert master["sum_open_interest"][0] == pytest.approx(1000.0)
