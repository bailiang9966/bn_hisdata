"""
Long-Period Multi-Timeframe Bounded Oscillator & Indicator Shape Feature Generator.
"""
import argparse
from pathlib import Path

import polars as pl

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def compute_stoch(df_lazy: pl.LazyFrame, n: int, k_smooth: int = 3, d_smooth: int = 3, prefix: str = "") -> pl.LazyFrame:
    low_n = pl.col("low").rolling_min(window_size=n)
    high_n = pl.col("high").rolling_max(window_size=n)
    stoch_k = ((pl.col("close") - low_n) / (high_n - low_n + 1e-8)) * 100.0
    stoch_k_smooth = stoch_k.rolling_mean(window_size=k_smooth)
    stoch_d = stoch_k_smooth.rolling_mean(window_size=d_smooth)

    k_name = f"{prefix}stoch_k_{n}"
    d_name = f"{prefix}stoch_d_{n}"

    return df_lazy.with_columns([
        stoch_k_smooth.alias(k_name),
        stoch_d.alias(d_name),
    ])


def compute_rsi(df_lazy: pl.LazyFrame, n: int, prefix: str = "") -> pl.LazyFrame:
    diff = pl.col("close") - pl.col("close").shift(1)
    gain = pl.when(diff > 0).then(diff).otherwise(0.0).rolling_mean(window_size=n)
    loss = pl.when(diff < 0).then(-diff).otherwise(0.0).rolling_mean(window_size=n)
    rs = gain / (loss + 1e-8)
    rsi_n = 100.0 - (100.0 / (1.0 + rs))
    return df_lazy.with_columns(rsi_n.alias(f"{prefix}rsi_{n}"))


def build_shape_feature_table(master_parquet_path: str, target_horizon: int = 5) -> pl.DataFrame:
    df_raw = pl.read_parquet(master_parquet_path).sort("close_time")

    # Add datetime_close for resampling
    lf = df_raw.lazy().with_columns(
        pl.col("close_time").cast(pl.Datetime("ms")).alias("datetime_close")
    )

    # 1. Compute 1m Long-Period Stoch (N = 30, 50, 80, 120, 200)
    stoch_periods_1m = [30, 50, 80, 120, 200]
    for n in stoch_periods_1m:
        lf = compute_stoch(lf, n, prefix="1m_")

    # 2. Compute 1m Long-Period RSI (N = 30, 60, 120)
    rsi_periods_1m = [30, 60, 120]
    for n in rsi_periods_1m:
        lf = compute_rsi(lf, n, prefix="1m_")

    # 3. Compute 5m Resampled Long-Period Stoch & RSI
    lf_5m = (
        lf.group_by_dynamic("datetime_close", every="5m", label="right")
        .agg([
            pl.col("open").first().alias("open_5m"),
            pl.col("high").max().alias("high_5m"),
            pl.col("low").min().alias("low_5m"),
            pl.col("close").last().alias("close_5m"),
            pl.col("close_time").last().alias("close_time"),
        ])
        .sort("close_time")
    )

    # Compute 5m Stoch (N = 10, 20, 40, 80 5m bars = 50m, 100m, 200m, 400m)
    for n in [10, 20, 40, 80]:
        low_n = pl.col("low_5m").rolling_min(window_size=n)
        high_n = pl.col("high_5m").rolling_max(window_size=n)
        stoch_k_5m = ((pl.col("close_5m") - low_n) / (high_n - low_n + 1e-8)) * 100.0
        stoch_k_smooth = stoch_k_5m.rolling_mean(window_size=3)
        stoch_d_smooth = stoch_k_smooth.rolling_mean(window_size=3)
        lf_5m = lf_5m.with_columns([
            stoch_k_smooth.alias(f"5m_stoch_k_{n}"),
            stoch_d_smooth.alias(f"5m_stoch_d_{n}"),
        ])

    # Compute 5m RSI (N = 14, 28, 50)
    for n in [14, 28, 50]:
        diff_5m = pl.col("close_5m") - pl.col("close_5m").shift(1)
        gain_5m = pl.when(diff_5m > 0).then(diff_5m).otherwise(0.0).rolling_mean(window_size=n)
        loss_5m = pl.when(diff_5m < 0).then(-diff_5m).otherwise(0.0).rolling_mean(window_size=n)
        rs_5m = gain_5m / (loss_5m + 1e-8)
        rsi_5m = 100.0 - (100.0 / (1.0 + rs_5m))
        lf_5m = lf_5m.with_columns(rsi_5m.alias(f"5m_rsi_{n}"))

    # Join 5m features back to 1m master table using join_asof
    df_1m = lf.collect()
    df_5m_feat = lf_5m.collect().sort("close_time")

    df_joined = df_1m.join_asof(
        df_5m_feat.drop(["datetime_close", "open_5m", "high_5m", "low_5m", "close_5m"]),
        on="close_time",
        strategy="backward"
    )

    lf_shape = df_joined.lazy()

    # 4. Shape & Pattern Derivative Features (Slopes, Curvature, Crossovers, Alignment)
    shape_exprs = []

    # 1m Stoch Shape Derivatives for N = 50 & 80
    for n in [50, 80]:
        k_col = pl.col(f"1m_stoch_k_{n}")
        d_col = pl.col(f"1m_stoch_d_{n}")

        shape_exprs.append((k_col - d_col).alias(f"1m_stoch_diff_{n}"))

        shape_exprs.append((k_col - k_col.shift(1)).alias(f"1m_stoch_slope1_{n}"))
        shape_exprs.append((k_col - k_col.shift(3)).alias(f"1m_stoch_slope3_{n}"))
        shape_exprs.append((k_col - k_col.shift(5)).alias(f"1m_stoch_slope5_{n}"))

        slope1 = k_col - k_col.shift(1)
        shape_exprs.append((slope1 - slope1.shift(1)).alias(f"1m_stoch_accel_{n}"))

        shape_exprs.append(
            pl.when((k_col > 80) & (slope1 > 0)).then(1.0)
            .when((k_col < 20) & (slope1 < 0)).then(-1.0)
            .otherwise(0.0)
            .alias(f"1m_stoch_breakout_{n}")
        )

    # 5m Stoch Shape Derivatives & Confluence Alignment
    for n_5m in [20, 40]:
        k_5m = pl.col(f"5m_stoch_k_{n_5m}")
        d_5m = pl.col(f"5m_stoch_d_{n_5m}")
        slope_5m = k_5m - k_5m.shift(1)

        shape_exprs.append((k_5m - d_5m).alias(f"5m_stoch_diff_{n_5m}"))
        shape_exprs.append(slope_5m.alias(f"5m_stoch_slope_{n_5m}"))

        k_1m = pl.col("1m_stoch_k_80")
        slope_1m = k_1m - k_1m.shift(1)
        shape_exprs.append(
            pl.when((slope_1m > 0) & (slope_5m > 0)).then(1.0)
            .when((slope_1m < 0) & (slope_5m < 0)).then(-1.0)
            .otherwise(0.0)
            .alias(f"confluence_1m5m_{n_5m}")
        )

    # Target: 5m future price direction
    future_close = pl.col("close").shift(-target_horizon)
    target_expr = pl.when(future_close > pl.col("close")).then(1).otherwise(0).alias("target")

    lf_shape = lf_shape.with_columns(shape_exprs + [target_expr])

    df_res = lf_shape.collect().drop("datetime_close")
    return df_res


def main():
    ap = argparse.ArgumentParser(description="Generate long-period multi-timeframe shape feature table")
    ap.add_argument("--master-path", default=str(DATA / "BTCUSDT_master_features_1m.parquet"))
    ap.add_argument("--output", default=str(DATA / "BTCUSDT_shape_features_1m.parquet"))
    args = ap.parse_args()

    df_shape = build_shape_feature_table(args.master_path)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_shape.write_parquet(out_path, compression="zstd")
    print(f"Successfully generated shape feature table: {out_path} ({len(df_shape)} rows, {len(df_shape.columns)} columns)")


if __name__ == "__main__":
    main()
