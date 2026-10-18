"""
Cumulative Volume Delta (CVD) and CVD Order Flow Derivative Feature Generator.

Features derived from active buying & selling volume:
  - Cumulative Volume Delta (CVD_um and CVD_spot)
  - Rolling CVD Momentum: delta_CVD over windows w in [1m, 3m, 5m, 15m, 30m, 60m, 240m]
  - CVD vs Price Divergence Signals:
      * Bullish Divergence: Price rolling low BUT CVD higher low (Absorption)
      * Bearish Divergence: Price rolling high BUT CVD lower high (Distribution)
  - CVD Robust Z-Score: (CVD - RollingMedian(CVD)) / (1.4826 * RollingMAD(CVD))
  - Futures CVD vs Spot CVD Spread: CVD_um - CVD_spot
  - CVD Acceleration: 2nd derivative of CVD
"""
import argparse
from pathlib import Path

import polars as pl

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def build_cvd_feature_table(master_parquet_path: str, target_horizon: int = 5) -> pl.DataFrame:
    df_raw = pl.read_parquet(master_parquet_path).sort("close_time")
    lf = df_raw.lazy()

    # 1. Base Net Volume Delta and CVD
    lf = lf.with_columns([
        (pl.col("um_buy_vol") - pl.col("um_sell_vol")).alias("um_net_delta"),
        (pl.col("spot_buy_vol") - pl.col("spot_sell_vol")).alias("spot_net_delta"),
    ]).with_columns([
        pl.col("um_net_delta").cum_sum().alias("cvd_um"),
        pl.col("spot_net_delta").cum_sum().alias("cvd_spot"),
    ])

    # Futures CVD vs Spot CVD Spread
    lf = lf.with_columns([
        (pl.col("cvd_um") - pl.col("cvd_spot")).alias("cvd_futures_spot_spread"),
    ])

    windows = [1, 3, 5, 15, 30, 60, 240]
    cvd_exprs = []

    # 2. Multi-Window CVD Momentum & Velocity
    for w in windows:
        # CVD Momentum
        cvd_exprs.append((pl.col("cvd_um") - pl.col("cvd_um").shift(w)).alias(f"cvd_mom_um_{w}m"))
        cvd_exprs.append((pl.col("cvd_spot") - pl.col("cvd_spot").shift(w)).alias(f"cvd_mom_spot_{w}m"))

        # CVD Acceleration
        mom = pl.col("cvd_um") - pl.col("cvd_um").shift(w)
        cvd_exprs.append((mom - mom.shift(w)).alias(f"cvd_accel_um_{w}m"))

        # CVD Robust Z-Score
        if w in [15, 60, 240]:
            r_med = pl.col("cvd_um").rolling_median(window_size=w)
            r_mad = (pl.col("cvd_um") - r_med).abs().rolling_median(window_size=w)
            cvd_exprs.append(((pl.col("cvd_um") - r_med) / (1.4826 * r_mad + 1e-8)).alias(f"cvd_robust_zscore_{w}m"))

        # 3. CVD - Price Divergence Indicators
        if w in [15, 30, 60]:
            # Price rolling min/max vs CVD rolling min/max
            p_min = pl.col("close").rolling_min(window_size=w)
            p_max = pl.col("close").rolling_max(window_size=w)
            c_min = pl.col("cvd_um").rolling_min(window_size=w)
            c_max = pl.col("cvd_um").rolling_max(window_size=w)

            # Bullish Divergence: Price is near rolling low BUT CVD is significantly above rolling low
            bull_div = pl.when((pl.col("close") <= p_min * 1.0005) & (pl.col("cvd_um") > c_min)).then(1.0).otherwise(0.0)
            # Bearish Divergence: Price is near rolling high BUT CVD is significantly below rolling high
            bear_div = pl.when((pl.col("close") >= p_max * 0.9995) & (pl.col("cvd_um") < c_max)).then(1.0).otherwise(0.0)

            cvd_exprs.append(bull_div.alias(f"cvd_bull_div_{w}m"))
            cvd_exprs.append(bear_div.alias(f"cvd_bear_div_{w}m"))

    # Target: 5m future price direction
    future_close = pl.col("close").shift(-target_horizon)
    target_expr = pl.when(future_close > pl.col("close")).then(1).otherwise(0).alias("target")

    lf_cvd = lf.with_columns(cvd_exprs + [target_expr])
    df_res = lf_cvd.collect()
    return df_res


def main():
    ap = argparse.ArgumentParser(description="Generate CVD and Order Flow Derivative Features")
    ap.add_argument("--master-path", default=str(DATA / "BTCUSDT_master_features_1m.parquet"))
    ap.add_argument("--output", default=str(DATA / "BTCUSDT_cvd_features_1m.parquet"))
    args = ap.parse_args()

    df_cvd = build_cvd_feature_table(args.master_path)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_cvd.write_parquet(out_path, compression="zstd")
    print(f"Successfully generated CVD feature table: {out_path} ({len(df_cvd)} rows, {len(df_cvd.columns)} columns)")


if __name__ == "__main__":
    main()
