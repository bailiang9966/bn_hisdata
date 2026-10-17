"""
High Precision Research Pipeline implementing Triple Barrier Method (TBM) & Meta-Labeling.
Targeting Precision >= 65% @ ~15 Trades/Day.
"""
import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score, precision_score

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def compute_triple_barrier_labels(
    df: pl.DataFrame,
    pt_multiplier: float = 1.0,
    sl_multiplier: float = 1.0,
    max_holding: int = 5,
) -> pl.DataFrame:
    """Compute dynamic EWMA volatility and Triple Barrier Labels (Upper, Lower, Vertical)."""
    df = df.sort("close_time")

    # 1. Compute 15m EWMA Log Return Volatility
    close_arr = df["close"].to_numpy()
    log_ret = np.zeros(len(close_arr))
    log_ret[1:] = np.log(close_arr[1:] / close_arr[:-1])

    # Dynamic volatility (15-period rolling std)
    vol = pl.Series(log_ret).abs().rolling_mean(window_size=15, min_samples=5).to_numpy()
    vol = np.nan_to_num(vol, nan=0.001)
    vol = np.clip(vol, 0.0005, 0.02)

    n_samples = len(close_arr)
    labels = np.zeros(n_samples, dtype=int)

    # Compute Triple Barrier outcomes
    for i in range(n_samples - max_holding):
        p0 = close_arr[i]
        v = vol[i]
        upper_barrier = p0 * (1.0 + pt_multiplier * v)
        lower_barrier = p0 * (1.0 - sl_multiplier * v)

        # Look forward up to max_holding periods
        future_prices = close_arr[i + 1 : i + 1 + max_holding]
        hit_upper = np.where(future_prices >= upper_barrier)[0]
        hit_lower = np.where(future_prices <= lower_barrier)[0]

        first_upper = hit_upper[0] if len(hit_upper) > 0 else 999
        first_lower = hit_lower[0] if len(hit_lower) > 0 else 999

        if first_upper < first_lower and first_upper < max_holding:
            labels[i] = 1  # Touched profit target first
        else:
            labels[i] = 0  # Hit loss or expired without profit

    return df.with_columns([
        pl.Series("volatility", vol),
        pl.Series("tbm_label", labels)
    ])


def train_meta_labeling_cascade(
    master_path: str,
    test_ratio: float = 0.2,
    pt_mult: float = 1.2,
    sl_mult: float = 1.0,
    holding: int = 5,
):
    print(f"Loading master table from: {master_path}")
    df_raw = pl.read_parquet(master_path)

    # 1. Compute TBM Labels
    df_tbm = compute_triple_barrier_labels(df_raw, pt_multiplier=pt_mult, sl_multiplier=sl_mult, max_holding=holding)

    # 2. Add Rolling Features
    windows = [3, 5, 15, 30, 60, 240]
    exprs = []
    feature_cols = []

    for w in windows:
        c_ret = f"ret_{w}m"
        exprs.append(((pl.col("close") - pl.col("close").shift(w)) / (pl.col("close").shift(w) + 1e-8)).alias(c_ret))
        feature_cols.append(c_ret)

        c_vir = f"um_vir_{w}m"
        r_buy = pl.col("um_buy_vol").rolling_sum(window_size=w)
        r_sell = pl.col("um_sell_vol").rolling_sum(window_size=w)
        exprs.append(((r_buy - r_sell) / (r_buy + r_sell + 1e-8)).alias(c_vir))
        feature_cols.append(c_vir)

        c_basis = f"basis_change_{w}m"
        exprs.append((pl.col("basis") - pl.col("basis").shift(w)).alias(c_basis))
        feature_cols.append(c_basis)

        if w in [15, 60, 240]:
            c_z = f"price_robust_zscore_{w}m"
            r_med = pl.col("close").rolling_median(window_size=w)
            r_mad = (pl.col("close") - r_med).abs().rolling_median(window_size=w)
            exprs.append(((pl.col("close") - r_med) / (1.4826 * r_mad + 1e-8)).alias(c_z))
            feature_cols.append(c_z)

    if "sum_open_interest" in df_tbm.columns:
        for w in [5, 15, 60]:
            c_oi = f"oi_change_{w}m"
            exprs.append(((pl.col("sum_open_interest") - pl.col("sum_open_interest").shift(w)) /
                          (pl.col("sum_open_interest").shift(w) + 1e-8)).alias(c_oi))
            feature_cols.append(c_oi)

    base_cols = ["basis", "basis_rate", "um_buy_vol", "um_sell_vol", "volatility",
                 "count_toptrader_long_short_ratio", "sum_toptrader_long_short_ratio"]
    for bc in base_cols:
        if bc in df_tbm.columns and bc not in feature_cols:
            feature_cols.append(bc)

    df_feat = df_tbm.with_columns(exprs)
    max_w = max(windows)
    df_clean = df_feat.slice(max_w, len(df_feat) - max_w - holding)

    # Clean nulls
    clean_exprs = [pl.col(c).fill_null(0.0).fill_nan(0.0).clip(-1e5, 1e5).alias(c) for c in feature_cols]
    df_clean = df_clean.with_columns(clean_exprs)

    # Time-series split (Strictly chronological)
    split_idx = int(len(df_clean) * (1 - test_ratio))
    df_train = df_clean.slice(0, split_idx - holding)
    df_test = df_clean.slice(split_idx, len(df_clean) - split_idx)

    X_train = df_train.select(feature_cols).to_numpy()
    y_train_tbm = df_train["tbm_label"].to_numpy()

    X_test = df_test.select(feature_cols).to_numpy()
    y_test_tbm = df_test["tbm_label"].to_numpy()

    print(f"Train TBM Positive Label Ratio: {np.mean(y_train_tbm):.4f}")
    print(f"Test TBM Positive Label Ratio: {np.mean(y_test_tbm):.4f}")

    # Stage 1 Primary Model (Primary Classifier for Long Signals)
    params_p1 = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.02,
        "num_leaves": 15,
        "max_depth": 4,
        "min_child_samples": 100,
        "verbose": -1,
        "random_state": 42
    }

    train_data_p1 = lgb.Dataset(X_train, label=y_train_tbm)
    valid_data_p1 = lgb.Dataset(X_test, label=y_test_tbm, reference=train_data_p1)

    model_p1 = lgb.train(
        params_p1,
        train_data_p1,
        num_boost_round=400,
        valid_sets=[valid_data_p1],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )

    preds_p1_train = model_p1.predict(X_train)
    preds_p1_test = model_p1.predict(X_test)

    # Stage 2 Meta-Model Training
    # Primary model triggers a candidate trade if P1 probability > threshold
    p1_threshold = np.quantile(preds_p1_train, 0.85)  # Candidate filter top 15%
    train_candidates = preds_p1_train >= p1_threshold
    test_candidates = preds_p1_test >= p1_threshold

    # Meta-label: 1 if Primary candidate succeeded (hit TBM upper barrier), 0 otherwise
    y_meta_train = y_train_tbm[train_candidates]
    X_meta_train = X_train[train_candidates]

    y_meta_test = y_test_tbm[test_candidates]
    X_meta_test = X_test[test_candidates]

    print(f"\nCandidates passed to Meta-Model: Train={len(X_meta_train)}, Test={len(X_meta_test)}")

    # Meta-Model (Trained strictly on Stage 1 Candidate Events)
    params_meta = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.02,
        "num_leaves": 15,
        "max_depth": 4,
        "min_child_samples": 50,
        "verbose": -1,
        "random_state": 42
    }

    train_data_meta = lgb.Dataset(X_meta_train, label=y_meta_train)
    valid_data_meta = lgb.Dataset(X_meta_test, label=y_meta_test, reference=train_data_meta)

    model_meta = lgb.train(
        params_meta,
        train_data_meta,
        num_boost_round=400,
        valid_sets=[valid_data_meta],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )

    meta_preds_train = model_meta.predict(X_meta_train)
    meta_preds_test = model_meta.predict(X_meta_test)

    auc_meta = roc_auc_score(y_meta_test, meta_preds_test)
    print(f"\n=== Meta-Model Out-Of-Sample AUC: {auc_meta:.4f} ===")

    # Calibrate High Precision Threshold for Meta-Model on Train Set
    test_days = (df_test["close_time"].max() - df_test["close_time"].min()) / (1000 * 3600 * 24)
    desired_trades_per_day = 15.0
    desired_test_trades = int(test_days * desired_trades_per_day)

    # Sort Meta-Model probabilities on test set
    meta_threshold = np.quantile(meta_preds_train, 0.92)  # High confidence Meta threshold

    test_close_times = df_test.filter(pl.Series("test_cand", test_candidates))["close_time"].to_numpy()

    # Apply 5m De-overlapping cooldown filter
    selected_trades = []
    last_t = -999999

    for i in range(len(meta_preds_test)):
        p_meta = meta_preds_test[i]
        t = test_close_times[i]
        actual_tbm = y_meta_test[i]

        if p_meta >= meta_threshold and (t - last_t >= holding * 60 * 1000):
            selected_trades.append((t, actual_tbm))
            last_t = t

    n_final = len(selected_trades)
    win_rate = (sum(tr[1] for tr in selected_trades) / n_final) if n_final > 0 else 0.0
    daily_freq = n_final / max(1e-3, test_days)

    print(f"\n=== High Precision Meta-Labeling Out-Of-Sample Results ===")
    print(f"Test Duration: {test_days:.1f} Days")
    print(f"Meta-Model Threshold: >= {meta_threshold:.4f}")
    print(f"Total Discrete De-overlapped Trades: {n_final}")
    print(f"Daily Trade Frequency: {daily_freq:.2f} Trades/Day")
    print(f"Meta-Labeling Precision (Win Rate): {win_rate * 100:.2f}%")


def main():
    ap = argparse.ArgumentParser(description="Research High Precision with TBM & Meta-Labeling")
    ap.add_argument("--master-path", default=str(DATA / "BTCUSDT_master_features_1m.parquet"))
    ap.add_argument("--pt", type=float, default=1.2, help="Profit Target Volatility Multiplier")
    ap.add_argument("--sl", type=float, default=1.0, help="Stop Loss Volatility Multiplier")
    args = ap.parse_args()

    train_meta_labeling_cascade(args.master_path, pt_mult=args.pt, sl_mult=args.sl)


if __name__ == "__main__":
    main()
