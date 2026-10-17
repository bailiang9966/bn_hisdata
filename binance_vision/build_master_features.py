"""
Build consolidated Master Feature Parquet table for a symbol (e.g., BTCUSDT).

Combines:
  1. Futures 1m Klines (OHLC + Volume + Counts)
  2. Spot 1m Klines (Volume + Counts + Spot Close for Basis)
  3. Futures 1m aggTrades Microstructure Features (VWAP disparity, Robust Z-score, RV, Sweep Depth, etc.)
  4. Futures 5m Metrics (Open Interest, Long/Short ratios via join_asof / ffill)
  5. Derives Basis (Futures Close - Spot Close)
  6. Provides rolling_quantile_bucket helper for rolling quantile bucket discretizations.
"""
import argparse
from pathlib import Path

import polars as pl

from binance_vision.build_aggtrade_features import compute_aggtrade_features

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def rolling_quantile_bucket(col_name: str, window_size: int, n_buckets: int = 10, alias: str = None) -> pl.Expr:
    """Compute integer quantile bucket (0 .. n_buckets-1) for a column over a rolling window_size."""
    if alias is None:
        alias = f"{col_name}_bucket_{window_size}m_q{n_buckets}"

    col = pl.col(col_name)
    r_min = col.rolling_min(window_size=window_size)
    r_max = col.rolling_max(window_size=window_size)

    # Scale to [0, 1] relative position within the rolling window, then map to Int8 bucket index (0..n_buckets-1)
    rel_pos = (col - r_min) / (r_max - r_min + 1e-8)
    bucket = (rel_pos * n_buckets).floor().cast(pl.Int8)
    bucket_clipped = pl.when(bucket >= n_buckets).then(n_buckets - 1).when(bucket < 0).then(0).otherwise(bucket)

    return bucket_clipped.cast(pl.Int8).alias(alias)


def build_master_feature_table(
    df_um_klines: pl.DataFrame | pl.LazyFrame,
    df_spot_klines: pl.DataFrame | pl.LazyFrame,
    df_aggtrades: pl.DataFrame | pl.LazyFrame | None = None,
    df_metrics: pl.DataFrame | pl.LazyFrame | None = None,
    timeframe: str = "1m",
) -> pl.DataFrame:
    """Combine futures/spot klines, aggtrades features, and metrics into a master feature table."""
    lf_um = df_um_klines.lazy() if isinstance(df_um_klines, pl.DataFrame) else df_um_klines
    lf_spot = df_spot_klines.lazy() if isinstance(df_spot_klines, pl.DataFrame) else df_spot_klines

    # 1. Futures Klines (Primary OHLC & Futures Volume)
    lf_um_prep = lf_um.with_columns(
        pl.when(pl.col("close_time") > 1e14)
          .then(pl.col("close_time") // 1000)
          .otherwise(pl.col("close_time"))
          .alias("close_time")
    ).select([
        pl.col("close_time"),
        pl.col("open"),
        pl.col("high"),
        pl.col("low"),
        pl.col("close"),
        pl.col("buy_vol").alias("um_buy_vol"),
        pl.col("sell_vol").alias("um_sell_vol"),
        pl.col("count").alias("um_count"),
    ])

    # 2. Spot Klines (Spot Volume & Spot Close for Basis)
    lf_spot_prep = lf_spot.with_columns(
        pl.when(pl.col("close_time") > 1e14)
          .then(pl.col("close_time") // 1000)
          .otherwise(pl.col("close_time"))
          .alias("close_time")
    ).select([
        pl.col("close_time"),
        pl.col("close").alias("spot_close"),
        pl.col("buy_vol").alias("spot_buy_vol"),
        pl.col("sell_vol").alias("spot_sell_vol"),
        pl.col("count").alias("spot_count"),
    ])

    # Deduplicate and Join Futures & Spot
    lf_um_prep = lf_um_prep.unique(subset=["close_time"])
    lf_spot_prep = lf_spot_prep.unique(subset=["close_time"])
    master_lf = lf_um_prep.join(lf_spot_prep, on="close_time", how="inner")

    # 3. Compute Basis (Futures Close - Spot Close)
    master_lf = master_lf.with_columns([
        (pl.col("close") - pl.col("spot_close")).alias("basis"),
    ])

    # 4. Join aggTrades Microstructure Features (if provided)
    if df_aggtrades is not None:
        lf_agg = compute_aggtrade_features(df_aggtrades, timeframe=timeframe)
        lf_agg = lf_agg.with_columns(
            pl.col("timestamp").dt.epoch("ms").alias("close_time")
        ).drop(["open", "high", "low", "close", "volume", "turnover", "timestamp"])

        master_lf = master_lf.join(lf_agg, on="close_time", how="left")

    master_df = master_lf.sort("close_time").collect()

    # 5. Join Metrics (5m sampling via join_asof / ffill) (if provided)
    if df_metrics is not None:
        df_met = df_metrics.lazy() if isinstance(df_metrics, pl.DataFrame) else df_metrics
        # Shift metrics +5m (300,000ms) to ensure no lookahead leakage
        df_met = df_met.select([
            (pl.col("create_time") + 300000).alias("create_time"),
            pl.col("sum_open_interest"),
            pl.col("sum_open_interest_value"),
            pl.col("count_toptrader_long_short_ratio"),
            pl.col("sum_toptrader_long_short_ratio"),
            pl.col("count_long_short_ratio"),
            pl.col("sum_taker_long_short_vol_ratio"),
        ]).sort("create_time").collect()

        master_df = master_df.join_asof(
            df_met,
            left_on="close_time",
            right_on="create_time",
            strategy="backward",
        ).drop("create_time")

    return master_df


def main():
    ap = argparse.ArgumentParser(description="Build consolidated Master Feature Parquet for a symbol")
    ap.add_argument("--symbol", default="BTCUSDT", help="Symbol (default: BTCUSDT)")
    ap.add_argument("--data-root", default=str(DATA), help="Root data directory")
    ap.add_argument("--output", default=None, help="Output master parquet file path")
    args = ap.parse_args()

    root = Path(args.data_root)
    symbol = args.symbol

    um_kline_files = sorted((root / "um" / "daily" / "klines" / symbol / "1m").glob("*.parquet")) + \
                     sorted((root / "um" / "monthly" / "klines" / symbol / "1m").glob("*.parquet"))
    spot_kline_files = sorted((root / "spot" / "daily" / "klines" / symbol / "1m").glob("*.parquet")) + \
                      sorted((root / "spot" / "monthly" / "klines" / symbol / "1m").glob("*.parquet"))

    if not um_kline_files or not spot_kline_files:
        print(f"Error: Missing kline files for {symbol} under {root}")
        return

    df_um = pl.read_parquet(um_kline_files)
    df_spot = pl.read_parquet(spot_kline_files)

    agg_files = sorted((root / "um" / "daily" / "aggTrades" / symbol).glob("*.parquet")) + \
                sorted((root / "um" / "monthly" / "aggTrades" / symbol).glob("*.parquet"))
    df_agg = pl.read_parquet(agg_files) if agg_files else None

    metric_files = sorted((root / "um" / "daily" / "metrics" / symbol).glob("*.parquet"))
    df_met = pl.read_parquet(metric_files) if metric_files else None

    master_df = build_master_feature_table(df_um, df_spot, df_agg, df_met, timeframe="1m")

    out_path = Path(args.output) if args.output else (root / f"{symbol}_master_features_1m.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    master_df.write_parquet(out_path, compression="zstd")
    print(f"Successfully generated master feature table: {out_path} ({len(master_df)} rows, {len(master_df.columns)} columns)")


if __name__ == "__main__":
    main()
