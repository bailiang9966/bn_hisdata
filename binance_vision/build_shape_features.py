"""
Focused Single Long-Period Stochastic Oscillator Trend & Shape Feature Builder.

Focuses purely on:
  - 1m Long-Period Stoch (N = 80)
  - 5m Resampled Long-Period Stoch (N = 40)
  - Pure Trend & Shape Derivatives: Slopes (1m, 3m, 5m, 10m), Acceleration (2nd derivative),
    Distance (K - D), and 1m/5m Trend Confluence alignment.
"""
import argparse
from pathlib import Path

import polars as pl

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def build_single_stoch_shape_table(master_parquet_path: str, target_horizon: int = 5) -> pl.DataFrame:
    df_raw = pl.read_parquet(master_parquet_path).sort("close_time")

    # Add datetime_close for resampling
    lf = df_raw.lazy().with_columns(
        pl.col("close_time").cast(pl.Datetime("ms")).alias("datetime_close")
    )

    # 1. 1m Long-Period Stoch (N = 80)
    low_80 = pl.col("low").rolling_min(window_size=80)
    high_80 = pl.col("high").rolling_max(window_size=80)
    stoch_k_1m = ((pl.col("close") - low_80) / (high_80 - low_80 + 1e-8)) * 100.0
    stoch_d_1m = stoch_k_1m.rolling_mean(window_size=3)

    lf = lf.with_columns([
        stoch_k_1m.alias("stoch_k_1m"),
        stoch_d_1m.alias("stoch_d_1m"),
    ])

    # 2. 5m Resampled Long-Period Stoch (N = 40)
    lf_5m = (
        lf.group_by_dynamic("datetime_close", every="5m", label="right")
        .agg([
            pl.col("low").min().alias("low_5m"),
            pl.col("high").max().alias("high_5m"),
            pl.col("close").last().alias("close_5m"),
            pl.col("close_time").last().alias("close_time"),
        ])
        .sort("close_time")
    )

    low_40_5m = pl.col("low_5m").rolling_min(window_size=40)
    high_40_5m = pl.col("high_5m").rolling_max(window_size=40)
    stoch_k_5m = ((pl.col("close_5m") - low_40_5m) / (high_40_5m - low_40_5m + 1e-8)) * 100.0
    stoch_d_5m = stoch_k_5m.rolling_mean(window_size=3)

    lf_5m = lf_5m.with_columns([
        stoch_k_5m.alias("stoch_k_5m"),
        stoch_d_5m.alias("stoch_d_5m"),
    ])

    # 3. Join 5m Stoch back to 1m table
    df_1m = lf.collect()
    df_5m_feat = lf_5m.collect().sort("close_time")

    df_joined = df_1m.join_asof(
        df_5m_feat.drop(["datetime_close", "low_5m", "high_5m", "close_5m"]),
        on="close_time",
        strategy="backward"
    )

    lf_shape = df_joined.lazy()

    # 4. Pure Trend Shape & Waveform Derivatives
    k1 = pl.col("stoch_k_1m")
    d1 = pl.col("stoch_d_1m")
    k5 = pl.col("stoch_k_5m")

    s1_1m = k1 - k1.shift(1)
    s3_1m = k1 - k1.shift(3)
    s5_1m = k1 - k1.shift(5)
    s10_1m = k1 - k1.shift(10)
    acc1_1m = s1_1m - s1_1m.shift(1)

    s1_5m = k5 - k5.shift(1)
    s5_5m = k5 - k5.shift(5)

    shape_exprs = [
        k1.alias("stoch_k_1m"),
        (k1 - d1).alias("stoch_diff_1m"),
        s1_1m.alias("stoch_slope1_1m"),
        s3_1m.alias("stoch_slope3_1m"),
        s5_1m.alias("stoch_slope5_1m"),
        s10_1m.alias("stoch_slope10_1m"),
        acc1_1m.alias("stoch_accel_1m"),
        k5.alias("stoch_k_5m"),
        s1_5m.alias("stoch_slope1_5m"),
        s5_5m.alias("stoch_slope5_5m"),
        # Multi-timeframe Trend Confluence (1m Slope > 0 AND 5m Slope > 0)
        pl.when((s1_1m > 0) & (s1_5m > 0)).then(1.0)
        .when((s1_1m < 0) & (s1_5m < 0)).then(-1.0)
        .otherwise(0.0).alias("stoch_trend_confluence")
    ]

    # Target: 5m future price direction
    future_close = pl.col("close").shift(-target_horizon)
    target_expr = pl.when(future_close > pl.col("close")).then(1).otherwise(0).alias("target")

    lf_shape = lf_shape.with_columns(shape_exprs + [target_expr])

    df_res = lf_shape.collect().drop("datetime_close")
    return df_res


def main():
    ap = argparse.ArgumentParser(description="Generate single long-period Stoch trend shape feature table")
    ap.add_argument("--master-path", default=str(DATA / "BTCUSDT_master_features_1m.parquet"))
    ap.add_argument("--output", default=str(DATA / "BTCUSDT_single_stoch_shape_1m.parquet"))
    args = ap.parse_args()

    df_shape = build_single_stoch_shape_table(args.master_path)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_shape.write_parquet(out_path, compression="zstd")
    print(f"Successfully generated single Stoch shape table: {out_path} ({len(df_shape)} rows, {len(df_shape.columns)} columns)")


if __name__ == "__main__":
    main()
