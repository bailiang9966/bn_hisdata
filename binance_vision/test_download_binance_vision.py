"""
Unit tests for binance_vision/download_binance_vision.py
"""
import hashlib
import io
import tempfile
import zipfile
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest
import requests

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

    path_metrics = dest_path(root, "um", "daily", "metrics", "BTCUSDT", "1m", "2025-06-01")
    assert path_metrics == Path("/tmp/data/um/daily/metrics/BTCUSDT/BTCUSDT-metrics-2025-06-01.zip")


def test_parse_list_arg():
    assert parse_list_arg("BTCUSDT, ETHUSDT ,SOLUSDT") == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    assert parse_list_arg("klines, metrics, ") == ["klines", "metrics"]


def test_get_schema_overrides():
    fdtype = pl.Float32
    schema_k = get_schema_overrides("klines", fdtype)
    assert schema_k["open"] == pl.Float32
    assert schema_k["open_time"] == pl.Int64

    schema_agg = get_schema_overrides("aggTrades", fdtype)
    assert schema_agg["is_buyer_maker"] == pl.Boolean

    schema_m = get_schema_overrides("metrics", fdtype)
    assert schema_m["sum_open_interest"] == pl.Float32


def create_zip_with_csv(csv_content: str, csv_filename: str = "data.csv") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(csv_filename, csv_content)
    return buf.getvalue()


def test_process_klines(tmp_path):
    # CSV content with header
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
    assert df["sell_vol"][0] == pytest.approx(6.0)  # volume 10.0 - taker_buy_volume 4.0
    assert df["open_time"].dtype == pl.Int64
    assert df["open"].dtype == pl.Float32


def test_process_klines_no_header(tmp_path):
    csv_no_header = "1700000000000,100.0,105.0,95.0,102.0,10.0,1700000059999,1020.0,50,4.0,408.0,0\n"
    zip_bytes = create_zip_with_csv(csv_no_header)
    zip_file = tmp_path / "test-kline-noheader.zip"
    zip_file.write_bytes(zip_bytes)

    logs = []
    process(zip_file, "klines", "parquet", pl.Float32, logs.append)

    parquet_out = zip_file.with_suffix(".parquet")
    assert parquet_out.exists()

    df = pl.read_parquet(parquet_out)
    assert len(df) == 1
    assert df["buy_vol"][0] == pytest.approx(4.0)


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


def test_process_metrics(tmp_path):
    csv_met = (
        "create_time,symbol,sum_open_interest,sum_open_interest_value,count_toptrader_long_short_ratio,sum_toptrader_long_short_ratio,count_long_short_ratio,sum_taker_long_short_vol_ratio\n"
        "2025-06-01 00:00:00,BTCUSDT,1000.0,500000.0,1.2,1.3,1.1,1.0\n"
    )
    zip_bytes = create_zip_with_csv(csv_met)
    zip_file = tmp_path / "test-metrics.zip"
    zip_file.write_bytes(zip_bytes)

    logs = []
    process(zip_file, "metrics", "parquet", pl.Float32, logs.append)

    parquet_out = zip_file.with_suffix(".parquet")
    assert parquet_out.exists()

    df = pl.read_parquet(parquet_out)
    assert len(df) == 1
    # Check "2025-06-01 00:00:00" converted to epoch ms Int64
    assert df["create_time"][0] == 1748736000000
    assert df["sum_open_interest"][0] == pytest.approx(1000.0)


def test_fetch_expected_sha256():
    session = MagicMock()

    # Successful checksum download
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = "A1B2C3D4E5F6  BTCUSDT-1m-2025-06-01.zip\n"
    session.get.return_value = mock_resp

    logs = []
    sha = fetch_expected_sha256(session, "http://example.com/file.zip.CHECKSUM", logs.append)
    assert sha == "a1b2c3d4e5f6"

    # HTTP 404 response
    mock_resp.status_code = 404
    sha_404 = fetch_expected_sha256(session, "http://example.com/file.zip.CHECKSUM", logs.append)
    assert sha_404 is None


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
