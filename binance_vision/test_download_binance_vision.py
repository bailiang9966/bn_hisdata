"""
Unit tests for binance_vision/download_binance_vision.py
"""
import hashlib
import io
import zipfile
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import polars as pl
import pytest

from binance_vision.download_binance_vision import (
    dest_path,
    download_one,
    fetch_expected_sha256,
    get_schema_overrides,
    parse_list_arg,
    period_strings,
    process,
    url_for,
)


def test_period_strings_daily():
    start = date(2025, 6, 1)
    end = date(2025, 6, 3)
    periods = list(period_strings("daily", start, end))
    assert periods == ["2025-06-01", "2025-06-02", "2025-06-03"]


def test_period_strings_monthly():
    start = date(2024, 11, 15)
    end = date(2025, 2, 10)
    periods = list(period_strings("monthly", start, end))
    assert periods == ["2024-11", "2024-12", "2025-01", "2025-02"]


def test_url_for():
    url_kline = url_for("spot", "daily", "klines", "BTCUSDT", "1m", "2025-06-01")
    assert url_kline == "https://data.binance.vision/data/spot/daily/klines/BTCUSDT/1m/BTCUSDT-1m-2025-06-01.zip"

    url_metrics = url_for("futures/um", "daily", "metrics", "BTCUSDT", "1m", "2025-06-01")
    assert url_metrics == "https://data.binance.vision/data/futures/um/daily/metrics/BTCUSDT/BTCUSDT-metrics-2025-06-01.zip"


def test_dest_path():
    root = Path("/tmp/data")
    path_kline = dest_path(root, "spot", "daily", "klines", "BTCUSDT", "1m", "2025-06-01")
    assert path_kline == Path("/tmp/data/spot/daily/klines/BTCUSDT/1m/BTCUSDT-1m-2025-06-01.zip")


def test_parse_list_arg():
    assert parse_list_arg("BTCUSDT, ETHUSDT ,SOLUSDT") == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


def test_get_schema_overrides():
    fdtype = pl.Float32
    schema_k = get_schema_overrides("klines", fdtype)
    assert schema_k["open"] == pl.Float32
    assert schema_k["open_time"] == pl.Int64


def create_zip_with_csv(csv_content: str, csv_filename: str = "data.csv") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(csv_filename, csv_content)
    return buf.getvalue()


def test_process_klines(tmp_path):
    csv_header = (
        "open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore\n"
        "1700000000000,100.0,105.0,95.0,102.0,10.0,1700000059999,1020.0,50,4.0,408.0,0\n"
    )
    zip_bytes = create_zip_with_csv(csv_header)
    zip_file = tmp_path / "test-kline-header.zip"
    zip_file.write_bytes(zip_bytes)

    logs = []
    process(zip_file, "klines", "parquet", pl.Float32, logs.append)

    parquet_out = zip_file.with_suffix(".parquet")
    assert parquet_out.exists()

    df = pl.read_parquet(parquet_out)
    assert df.columns == ["open_time", "open", "high", "low", "close", "close_time", "count", "buy_vol", "sell_vol"]
    assert df["buy_vol"][0] == pytest.approx(4.0)
    assert df["sell_vol"][0] == pytest.approx(6.0)
    assert df["close_time"][0] == 1700000060000  # raw close_time (1700000059999) + 1ms


def test_process_aggtrades(tmp_path):
    csv_agg = (
        "agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker,is_best_match\n"
        "1001,50000.5,0.02,2001,2002,1700000000000,true,true\n"
    )
    zip_bytes = create_zip_with_csv(csv_agg)
    zip_file = tmp_path / "test-aggtrades.zip"
    zip_file.write_bytes(zip_bytes)

    logs = []
    process(zip_file, "aggTrades", "parquet", pl.Float32, logs.append)

    parquet_out = zip_file.with_suffix(".parquet")
    assert parquet_out.exists()

    df = pl.read_parquet(parquet_out)
    assert len(df) == 1
    assert df["agg_trade_id"][0] == 1001
    assert df["is_buyer_maker"][0] is True


def test_fetch_expected_sha256():
    session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = "A1B2C3D4E5F6  BTCUSDT-1m-2025-06-01.zip\n"
    session.get.return_value = mock_resp

    logs = []
    sha = fetch_expected_sha256(session, "http://example.com/file.zip.CHECKSUM", logs.append)
    assert sha == "a1b2c3d4e5f6"


def test_download_one(tmp_path):
    session = MagicMock()
    data_bytes = b"sample zip content"
    expected_sha = hashlib.sha256(data_bytes).hexdigest()

    mock_checksum_resp = MagicMock()
    mock_checksum_resp.status_code = 200
    mock_checksum_resp.text = f"{expected_sha}  file.zip\n"

    mock_data_resp = MagicMock()
    mock_data_resp.status_code = 200
    mock_data_resp.iter_content.return_value = [data_bytes]
    mock_data_resp.__enter__.return_value = mock_data_resp

    def session_get_side_effect(url, **kwargs):
        if url.endswith(".CHECKSUM"):
            return mock_checksum_resp
        return mock_data_resp

    session.get.side_effect = session_get_side_effect

    dest = tmp_path / "sample.zip"
    logs = []
    res = download_one(session, "http://example.com/sample.zip", dest, True, logs.append)

    assert res == "ok"
    assert dest.exists()
    assert dest.read_bytes() == data_bytes


def test_compute_aggtrade_features():
    from binance_vision.build_aggtrade_features import compute_aggtrade_features

    df_raw = pl.DataFrame({
        "agg_trade_id": [1, 2, 3],
        "price": [100.0, 101.0, 100.5],
        "quantity": [1.0, 2.0, 0.5],
        "first_trade_id": [10, 20, 30],
        "last_trade_id": [10, 22, 30],
        "transact_time": [1700000000000, 1700000010000, 1700000020000],
        "is_buyer_maker": [False, True, False],
        "is_best_match": [True, True, True],
    })

    feat_df = compute_aggtrade_features(df_raw, "1m").collect()
    assert len(feat_df) == 1
    assert "vwap_disparity" in feat_df.columns
    assert "price_robust_zscore" in feat_df.columns
    assert "qty_iqr" in feat_df.columns
    assert "mean_sweep_depth" in feat_df.columns
    assert feat_df["trade_count"][0] == 3
