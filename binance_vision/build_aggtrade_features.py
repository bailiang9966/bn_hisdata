"""
Extract robust microstructure features from Binance aggTrades Parquet files.

Features extracted per window (default 1m or 5s):
  - VWAP disparity: (close - vwap) / vwap
  - Price Median & MAD (Median Absolute Deviation)
  - Robust Price Z-Score: (close - price_median) / (1.4826 * price_mad)
  - Realized Volatility (RV) & Up/Down Semi-Variance Ratio
  - Sweep Depth (mean_sweep_depth, max_sweep_depth)
  - Trade Quantity Robust Statistics: Median, Q25, Q75, IQR, Q90/Median Tail Ratio
  - Trade Intensity: trade_count
  - Inter-Arrival Time: dt_mean, dt_std
"""
import argparse
from pathlib import Path

import polars as pl

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def compute_aggtrade_features(df_or_lf: pl.DataFrame | pl.LazyFrame, timeframe: str = "1m") -> pl.LazyFrame:
    """Compute robust microstructure features from an aggTrades frame."""
    lf = df_or_lf.lazy() if isinstance(df_or_lf, pl.DataFrame) else df_or_lf

    # 1. Base derived columns
    lf = lf.with_columns([
        (pl.col("price") * pl.col("quantity")).alias("notional"),
        (pl.col("last_trade_id") - pl.col("first_trade_id") + 1).alias("sweep_depth"),
        pl.col("transact_time").cast(pl.Datetime("ms")).alias("timestamp"),
    ]).sort("timestamp")

    # Log returns for Realized Volatility
    lf = lf.with_columns([
        (pl.col("price") / pl.col("price").shift(1)).log().alias("log_return"),
        (pl.col("transact_time") - pl.col("transact_time").shift(1)).alias("dt_ms"),
    ])

    # 2. Resample and aggregate per timeframe window (label="right" so window timestamp matches close_time)
    features_lf = lf.group_by_dynamic("timestamp", every=timeframe, label="right").agg([
        # OHLC & Close
        pl.col("price").first().alias("open"),
        pl.col("price").max().alias("high"),
        pl.col("price").min().alias("low"),
        pl.col("price").last().alias("close"),

        # Total Volume & Turnover & VWAP
        pl.col("quantity").sum().alias("volume"),
        pl.col("notional").sum().alias("turnover"),
        (pl.col("notional").sum() / (pl.col("quantity").sum() + 1e-8)).alias("vwap"),

        # Price Robust Statistics (Median & MAD)
        pl.col("price").median().alias("price_median"),
        (pl.col("price") - pl.col("price").median()).abs().median().alias("price_mad"),

        # Quantity Robust Statistics (Median, IQR, Q90)
        pl.col("quantity").median().alias("qty_median"),
        pl.col("quantity").quantile(0.25).alias("qty_q25"),
        pl.col("quantity").quantile(0.75).alias("qty_q75"),
        pl.col("quantity").quantile(0.90).alias("qty_q90"),

        # Realized Volatility & Semi-Variance
        (pl.col("log_return").pow(2).sum() + 1e-12).sqrt().alias("rv"),
        (pl.col("log_return").filter(pl.col("log_return") > 0).pow(2).sum() + 1e-12).sqrt().alias("rv_up"),
        (pl.col("log_return").filter(pl.col("log_return") < 0).pow(2).sum() + 1e-12).sqrt().alias("rv_down"),

        # Sweep Depth & Trade Counts
        pl.len().alias("trade_count"),
        pl.col("sweep_depth").mean().alias("mean_sweep_depth"),
        pl.col("sweep_depth").max().alias("max_sweep_depth"),

        # Inter-arrival Time (ms)
        pl.col("dt_ms").mean().alias("dt_mean"),
        pl.col("dt_ms").std().alias("dt_std"),
    ]).with_columns([
        # Secondary Derived Robust Indicators
        ((pl.col("close") - pl.col("vwap")) / (pl.col("vwap") + 1e-8)).alias("vwap_disparity"),
        ((pl.col("close") - pl.col("price_median")) / (1.4826 * pl.col("price_mad") + 1e-8)).alias("price_robust_zscore"),
        (pl.col("qty_q75") - pl.col("qty_q25")).alias("qty_iqr"),
        (pl.col("qty_q90") / (pl.col("qty_median") + 1e-8)).alias("qty_tail_ratio"),
        (pl.col("rv_up") / (pl.col("rv_down") + 1e-8)).alias("semi_var_ratio"),
    ])

    return features_lf


def process_file(file_path: Path, dest_dir: Path, timeframe: str = "1m") -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_path = dest_dir / f"{file_path.stem}_feat_{timeframe}.parquet"
    if out_path.exists():
        return out_path

    lf = compute_aggtrade_features(pl.scan_parquet(file_path), timeframe=timeframe)
    tmp_out = dest_dir / (out_path.name + ".part")
    lf.sink_parquet(str(tmp_out), compression="zstd")
    tmp_out.replace(out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Extract robust microstructure features from aggTrades")
    ap.add_argument("--input", required=True, help="Input aggTrades parquet file or directory")
    ap.add_argument("--output-dir", default=None, help="Output directory (defaults to input dir/features)")
    ap.add_argument("--timeframe", default="1m", help="Resample window e.g. 1m, 5s, 10s")
    args = ap.parse_args()

    inp = Path(args.input)
    if inp.is_file():
        files = [inp]
    else:
        files = list(inp.rglob("*.parquet"))

    out_dir = Path(args.output_dir) if args.output_dir else (inp.parent / "features" if inp.is_file() else inp / "features")

    for f in files:
        if "_feat_" in f.name:
            continue
        out = process_file(f, out_dir, args.timeframe)
        print(f"Processed: {f.name} -> {out.name}", flush=True)


if __name__ == "__main__":
    main()
