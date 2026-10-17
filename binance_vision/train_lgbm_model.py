"""
LightGBM High-Confidence 5-Minute Event Contract Model Trainer.
"""
import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score, log_loss

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def add_rolling_features_and_targets(df: pl.DataFrame, target_horizon: int = 5) -> tuple[pl.DataFrame, list[str]]:
    """Compute strictly backward-looking rolling features and target without lookahead bias."""
    df = df.sort("close_time")

    windows = [3, 5, 15, 30, 60, 240]
    exprs = []
    feature_cols = []

    for w in windows:
        # Returns
        col_ret = f"ret_{w}m"
        exprs.append(((pl.col("close") - pl.col("close").shift(w)) / (pl.col("close").shift(w) + 1e-8)).alias(col_ret))
        feature_cols.append(col_ret)

        # Volume Imbalance Ratio (VIR)
        col_vir = f"um_vir_{w}m"
        r_buy = pl.col("um_buy_vol").rolling_sum(window_size=w)
        r_sell = pl.col("um_sell_vol").rolling_sum(window_size=w)
        exprs.append(((r_buy - r_sell) / (r_buy + r_sell + 1e-8)).alias(col_vir))
        feature_cols.append(col_vir)

        # Basis Change
        col_basis = f"basis_change_{w}m"
        exprs.append((pl.col("basis") - pl.col("basis").shift(w)).alias(col_basis))
        feature_cols.append(col_basis)

        # Robust Z-Score
        if w in [15, 60, 240]:
            col_z = f"price_robust_zscore_{w}m"
            r_med = pl.col("close").rolling_median(window_size=w)
            r_mad = (pl.col("close") - r_med).abs().rolling_median(window_size=w)
            exprs.append(((pl.col("close") - r_med) / (1.4826 * r_mad + 1e-8)).alias(col_z))
            feature_cols.append(col_z)

    # OI Changes
    for w in [5, 15, 60]:
        col_oi = f"oi_change_{w}m"
        exprs.append(((pl.col("sum_open_interest") - pl.col("sum_open_interest").shift(w)) /
                      (pl.col("sum_open_interest").shift(w) + 1e-8)).alias(col_oi))
        feature_cols.append(col_oi)

    # Add existing columns to feature_cols
    base_features = [
        "basis", "basis_rate", "um_buy_vol", "um_sell_vol", "um_count",
        "spot_buy_vol", "spot_sell_vol", "spot_count",
        "sum_open_interest", "count_toptrader_long_short_ratio",
        "sum_toptrader_long_short_ratio", "count_long_short_ratio", "sum_taker_long_short_vol_ratio"
    ]
    for bf in base_features:
        if bf in df.columns and bf not in feature_cols:
            feature_cols.append(bf)

    # Calculate target (5m future price direction)
    future_close = pl.col("close").shift(-target_horizon)
    target_expr = pl.when(future_close > pl.col("close")).then(1).otherwise(0).alias("target")

    df_featured = df.with_columns(exprs + [target_expr])

    # Slice off edge nulls from shift / rolling
    max_w = max(windows)
    df_clean = df_featured.slice(max_w, len(df_featured) - max_w - target_horizon)

    return df_clean, feature_cols


def clean_dataframe(df: pl.DataFrame, feature_cols: list[str]) -> tuple[pl.DataFrame, list[str]]:
    """Clean nulls, NaNs, and Infs in feature columns."""
    valid_cols = []
    for c in feature_cols:
        null_cnt = df[c].null_count()
        if null_cnt < len(df) * 0.2:  # Keep columns with <20% missing
            valid_cols.append(c)

    # Fill nulls and clip extreme values in valid feature columns
    clean_exprs = []
    for c in valid_cols:
        clean_exprs.append(
            pl.col(c).fill_null(0.0).fill_nan(0.0).clip(-1e5, 1e5).alias(c)
        )

    df_cleaned = df.with_columns(clean_exprs).filter(pl.col("target").is_not_null())
    return df_cleaned, valid_cols


def train_and_evaluate(master_parquet_path: str, target_horizon: int = 5, test_ratio: float = 0.2):
    print(f"Loading master feature table from: {master_parquet_path}")
    df_master = pl.read_parquet(master_parquet_path)
    print(f"Loaded master table with {len(df_master)} rows.")

    df_featured, raw_feature_cols = add_rolling_features_and_targets(df_master, target_horizon)
    df_clean, feature_cols = clean_dataframe(df_featured, raw_feature_cols)

    print(f"Cleaned dataset rows: {len(df_clean)}, features count: {len(feature_cols)}")

    # Time-series split (Strict chronological order)
    split_idx = int(len(df_clean) * (1 - test_ratio))
    df_train = df_clean.slice(0, split_idx)
    df_test = df_clean.slice(split_idx, len(df_clean) - split_idx)

    print(f"Train set: {len(df_train)} rows, Test set: {len(df_test)} rows")

    X_train = df_train.select(feature_cols).to_numpy()
    y_train = df_train["target"].to_numpy()

    X_test = df_test.select(feature_cols).to_numpy()
    y_test = df_test["target"].to_numpy()

    # Train LightGBM Model
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbose": -1,
        "random_state": 42,
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[valid_data],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    preds_prob = model.predict(X_test)

    auc = roc_auc_score(y_test, preds_prob)
    loss = log_loss(y_test, preds_prob)
    print(f"\n=== Overall Test Metrics ===")
    print(f"AUC: {auc:.4f}")
    print(f"LogLoss: {loss:.4f}")

    # Evaluate High-Confidence Trades (~15-20 trades/day)
    test_days = (df_test["close_time"].max() - df_test["close_time"].min()) / (1000 * 3600 * 24)
    target_trades_per_day = 18.0
    desired_total_trades = max(10, int(test_days * target_trades_per_day))

    # Calibrate probability threshold
    dev_from_half = np.abs(preds_prob - 0.5)
    sorted_devs = np.sort(dev_from_half)[::-1]
    threshold_dev = sorted_devs[min(desired_total_trades, len(sorted_devs) - 1)]
    long_threshold = 0.5 + threshold_dev
    short_threshold = 0.5 - threshold_dev

    long_mask = preds_prob >= long_threshold
    short_mask = preds_prob <= short_threshold

    selected_mask = long_mask | short_mask
    selected_preds = (preds_prob[selected_mask] >= 0.5).astype(int)
    selected_y = y_test[selected_mask]

    n_trades = len(selected_y)
    win_rate = np.mean(selected_preds == selected_y) if n_trades > 0 else 0.0
    trades_per_day = n_trades / max(1e-3, test_days)

    print(f"\n=== High-Confidence Filtering Target: ~15-20 Trades/Day ===")
    print(f"Test Duration: {test_days:.1f} Days")
    print(f"Probability Thresholds: Long >= {long_threshold:.4f}, Short <= {short_threshold:.4f}")
    print(f"Total High-Confidence Trades: {n_trades}")
    print(f"Daily Trade Frequency: {trades_per_day:.2f} Trades/Day")
    print(f"High-Confidence Win Rate (Precision): {win_rate * 100:.2f}%")

    # Feature Importance Top 10
    imp = model.feature_importance(importance_type="gain")
    imp_indices = np.argsort(imp)[::-1][:10]

    print(f"\n=== Top 10 Feature Importances (Gain) ===")
    for idx in imp_indices:
        print(f"  {feature_cols[idx]:<30}: {imp[idx]:.2f}")


def main():
    ap = argparse.ArgumentParser(description="Train LightGBM model for 5m price prediction")
    ap.add_argument("--master-path", default=str(DATA / "BTCUSDT_master_features_1m.parquet"))
    ap.add_argument("--horizon", type=int, default=5, help="Target prediction horizon in minutes (default: 5)")
    args = ap.parse_args()

    train_and_evaluate(args.master_path, target_horizon=args.horizon)


if __name__ == "__main__":
    main()
