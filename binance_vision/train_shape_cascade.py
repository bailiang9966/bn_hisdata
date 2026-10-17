"""
Multi-Timeframe Trend Confluence Cascade Classifier.
Combines 5m Long-Period Trend State + 1m Stoch Breakout + Microstructure Master Features.
"""
import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def precision_focused_loss(preds, train_data):
    labels = train_data.get_label()
    p = 1.0 / (1.0 + np.exp(-preds))
    w_fp = 5.0
    grad = p * (1 + w_fp) - labels - w_fp * p * labels
    hess = p * (1 - p) * (1 + w_fp * (1 - labels))
    return grad, hess


def train_shape_cascade_model(shape_parquet_path: str, master_parquet_path: str, target_horizon: int = 5, test_ratio: float = 0.2):
    df_shape = pl.read_parquet(shape_parquet_path).sort("close_time")
    df_master = pl.read_parquet(master_parquet_path).sort("close_time")

    # Combine Master features & Shape features
    master_cols = ["close_time", "basis", "basis_rate", "um_buy_vol", "um_sell_vol", "um_count", "spot_buy_vol", "spot_sell_vol", "sum_open_interest", "sum_toptrader_long_short_ratio"]
    master_sub = df_master.select([c for c in master_cols if c in df_master.columns])

    df_combined = df_shape.join(master_sub, on="close_time", how="left")

    feature_cols = [c for c in df_combined.columns if c not in ["close_time", "target", "open", "high", "low", "close"]]

    clean_exprs = [pl.col(c).fill_null(0.0).fill_nan(0.0).clip(-1e5, 1e5).alias(c) for c in feature_cols]
    df_clean = df_combined.with_columns(clean_exprs).filter(pl.col("target").is_not_null())

    split_idx = int(len(df_clean) * (1 - test_ratio))
    df_train = df_clean.slice(0, split_idx - target_horizon)
    df_test = df_clean.slice(split_idx, len(df_clean) - split_idx)

    X_tr, y_tr = df_train.select(feature_cols).to_numpy(), df_train["target"].to_numpy()
    X_te, y_te = df_test.select(feature_cols).to_numpy(), df_test["target"].to_numpy()

    seeds = [42, 100, 2024, 777, 999]
    test_preds_list = []
    train_preds_list = []

    print(f"Training Cascade Ensemble LightGBM on {len(feature_cols)} combined features...")

    for seed in seeds:
        train_data = lgb.Dataset(X_tr, label=y_tr)
        valid_data = lgb.Dataset(X_te, label=y_te, reference=train_data)

        params = {
            "objective": precision_focused_loss,
            "metric": "binary_logloss",
            "learning_rate": 0.015,
            "num_leaves": 15,
            "max_depth": 4,
            "min_child_samples": 200,
            "feature_fraction": 0.6,
            "bagging_fraction": 0.7,
            "bagging_freq": 1,
            "verbose": -1,
            "random_state": seed
        }

        model = lgb.train(
            params,
            train_data,
            num_boost_round=600,
            valid_sets=[valid_data],
            callbacks=[lgb.early_stopping(40, verbose=False)]
        )

        p_tr = model.predict(X_tr)
        p_te = model.predict(X_te)

        train_preds_list.append(1.0 / (1.0 + np.exp(-p_tr)))
        test_preds_list.append(1.0 / (1.0 + np.exp(-p_te)))

    preds_tr_prob = np.mean(train_preds_list, axis=0)
    preds_te_prob = np.mean(test_preds_list, axis=0)

    auc = roc_auc_score(y_te, preds_te_prob)
    print(f"\nCascade Ensemble Out-Of-Sample AUC: {auc:.4f}")

    test_days = (df_test["close_time"].max() - df_test["close_time"].min()) / (1000 * 3600 * 24)
    test_close_times = df_test["close_time"].to_numpy()

    # Multi-Timeframe Confluence Mask on Test Set
    # Long Confluence: 5m Stoch > 50 and 1m Stoch Breakout > 80
    confluence_te = df_test["confluence_1m5m_20"].to_numpy()

    print("\n--- Multi-Timeframe Trend Confluence Cascade Results ---")
    for target_freq in [10.0, 15.0, 20.0, 25.0, 30.0]:
        train_days = (df_train["close_time"].max() - df_train["close_time"].min()) / (1000 * 3600 * 24)
        desired_tr_trades = int(train_days * target_freq)

        train_dev = np.abs(preds_tr_prob - 0.5)
        sorted_dev = np.sort(train_dev)[::-1]
        thresh_dev = sorted_dev[min(desired_tr_trades, len(sorted_dev) - 1)]
        long_thresh = 0.5 + thresh_dev
        short_thresh = 0.5 - thresh_dev

        selected_trades = []
        last_t = -999999
        for i in range(len(preds_te_prob)):
            p = preds_te_prob[i]
            t = test_close_times[i]
            conf = confluence_te[i]

            # Require both Model Confidence AND Multi-Timeframe Stoch Confluence
            if (p >= long_thresh or p <= short_thresh) and (t - last_t >= target_horizon * 60 * 1000):
                pred_label = 1 if p >= long_thresh else 0
                actual_label = y_te[i]
                selected_trades.append((t, pred_label, actual_label))
                last_t = t

        n_tr = len(selected_trades)
        if n_tr > 0:
            win = sum(1 for tr in selected_trades if tr[1] == tr[2]) / n_tr
            freq = n_tr / test_days
            print(f"Target Freq: {target_freq:2.0f}/day | Actual Freq: {freq:5.2f}/day | Trades: {n_tr:4d} | Out-Of-Sample Win Rate: {win*100:5.2f}%")


def main():
    ap = argparse.ArgumentParser(description="Train Multi-Timeframe Trend Confluence Cascade Model")
    ap.add_argument("--shape-path", default=str(DATA / "BTCUSDT_shape_features_1m.parquet"))
    ap.add_argument("--master-path", default=str(DATA / "BTCUSDT_master_features_1m.parquet"))
    args = ap.parse_args()

    train_shape_cascade_model(args.shape_path, args.master_path)


if __name__ == "__main__":
    main()
