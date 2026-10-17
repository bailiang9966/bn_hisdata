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

    # OI Changes (if OI exists)
    if "sum_open_interest" in df.columns:
        for w in [5, 15, 60]:
            col_oi = f"oi_change_{w}m"
            exprs.append(((pl.col("sum_open_interest") - pl.col("sum_open_interest").shift(w)) /
                          (pl.col("sum_open_interest").shift(w) + 1e-8)).alias(col_oi))
            feature_cols.append(col_oi)

    # Add existing base columns to feature_cols
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
        if null_cnt < len(df) * 0.2:
            valid_cols.append(c)

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

    # Time-series split (80% train, 20% test with 5m embargo to prevent overlap leakage)
    split_idx = int(len(df_clean) * (1 - test_ratio))
    df_train = df_clean.slice(0, split_idx - target_horizon)
    df_test = df_clean.slice(split_idx, len(df_clean) - split_idx)

    print(f"Train set: {len(df_train)} rows, Test set: {len(df_test)} rows")

    X_train = df_train.select(feature_cols).to_numpy()
    y_train = df_train["target"].to_numpy()

    X_test = df_test.select(feature_cols).to_numpy()
    y_test = df_test["target"].to_numpy()

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

    preds_test = model.predict(X_test)
    preds_train = model.predict(X_train)

    auc = roc_auc_score(y_test, preds_test)
    loss = log_loss(y_test, preds_test)
    print(f"\n=== Overall Test Metrics (Out-Of-Sample) ===")
    print(f"AUC: {auc:.4f}")
    print(f"LogLoss: {loss:.4f}")

    # Calibrate Probability Thresholds STRICTLY on Train Set (In-Sample Calibration)
    train_dev = np.abs(preds_train - 0.5)
    target_trades_per_day = 18.0
    train_days = (df_train["close_time"].max() - df_train["close_time"].min()) / (1000 * 3600 * 24)
    desired_train_trades = int(train_days * target_trades_per_day)

    sorted_train_dev = np.sort(train_dev)[::-1]
    thresh_dev = sorted_train_dev[min(desired_train_trades, len(sorted_train_dev) - 1)]
    long_threshold = 0.5 + thresh_dev
    short_threshold = 0.5 - thresh_dev

    print(f"\n=== Threshold Calibrated on Train Set ===")
    print(f"Long >= {long_threshold:.4f}, Short <= {short_threshold:.4f}")

    # Evaluate on Test Set with 5m De-overlapping cooldown filter
    test_days = (df_test["close_time"].max() - df_test["close_time"].min()) / (1000 * 3600 * 24)
    test_close_times = df_test["close_time"].to_numpy()

    selected_trades = []
    last_trade_time = -999999

    for i in range(len(preds_test)):
        p = preds_test[i]
        t = test_close_times[i]
        if (p >= long_threshold or p <= short_threshold) and (t - last_trade_time >= target_horizon * 60 * 1000):
            pred_label = 1 if p >= long_threshold else 0
            actual_label = y_test[i]
            selected_trades.append((t, pred_label, actual_label))
            last_trade_time = t

    n_trades = len(selected_trades)
    win_rate = (sum(1 for tr in selected_trades if tr[1] == tr[2]) / n_trades) if n_trades > 0 else 0.0
    trades_per_day = n_trades / max(1e-3, test_days)

    print(f"\n=== Out-Of-Sample Realistic De-Overlapped Results ===")
    print(f"Test Duration: {test_days:.1f} Days")
    print(f"Total Discrete Trades: {n_trades}")
    print(f"Daily Trade Frequency: {trades_per_day:.2f} Trades/Day")
    print(f"Out-Of-Sample Precision / Win Rate: {win_rate * 100:.2f}%")

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
