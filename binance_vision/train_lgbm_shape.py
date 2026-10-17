"""
LightGBM Classifier Trained on Long-Period Multi-Timeframe Stoch/RSI Shape Features.
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


def train_lgbm_shape_model(shape_parquet_path: str, target_horizon: int = 5, test_ratio: float = 0.2):
    df_shape = pl.read_parquet(shape_parquet_path).sort("close_time")

    feature_cols = [c for c in df_shape.columns if c not in ["close_time", "target", "open", "high", "low", "close"]]

    # Clean nulls
    clean_exprs = [pl.col(c).fill_null(0.0).fill_nan(0.0).clip(-1e5, 1e5).alias(c) for c in feature_cols]
    df_clean = df_shape.with_columns(clean_exprs).filter(pl.col("target").is_not_null())

    split_idx = int(len(df_clean) * (1 - test_ratio))
    df_train = df_clean.slice(0, split_idx - target_horizon)
    df_test = df_clean.slice(split_idx, len(df_clean) - split_idx)

    X_tr, y_tr = df_train.select(feature_cols).to_numpy(), df_train["target"].to_numpy()
    X_te, y_te = df_test.select(feature_cols).to_numpy(), df_test["target"].to_numpy()

    seeds = [42, 100, 2024, 777, 999]
    test_preds_list = []
    train_preds_list = []

    print(f"Training 5-Seed Ensemble LightGBM on {len(feature_cols)} Long-Period Shape Features...")

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
    print(f"\n5-Seed Shape Model Out-Of-Sample AUC: {auc:.4f}")

    test_days = (df_test["close_time"].max() - df_test["close_time"].min()) / (1000 * 3600 * 24)
    test_close_times = df_test["close_time"].to_numpy()

    print("\n--- Long-Period Multi-Timeframe Shape Model Results ---")
    for q in [0.90, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999]:
        thresh_dev = np.quantile(np.abs(preds_tr_prob - 0.5), q)
        long_thresh = 0.5 + thresh_dev
        short_thresh = 0.5 - thresh_dev

        selected_trades = []
        last_t = -999999
        for i in range(len(preds_te_prob)):
            p = preds_te_prob[i]
            t = test_close_times[i]
            if (p >= long_thresh or p <= short_thresh) and (t - last_t >= target_horizon * 60 * 1000):
                pred_label = 1 if p >= long_thresh else 0
                actual_label = y_te[i]
                selected_trades.append((t, pred_label, actual_label))
                last_t = t

        n_tr = len(selected_trades)
        if n_tr > 0:
            win = sum(1 for tr in selected_trades if tr[1] == tr[2]) / n_tr
            freq = n_tr / test_days
            print(f"Quantile {q*100:4.1f}% | Thresh: Long >= {long_thresh:.4f}, Short <= {short_thresh:.4f} | Trades: {n_tr:4d} | Freq: {freq:5.2f}/day | Win Rate: {win*100:5.2f}%")


def main():
    ap = argparse.ArgumentParser(description="Train LightGBM on Long-Period Multi-Timeframe Shape Features")
    ap.add_argument("--shape-path", default=str(DATA / "BTCUSDT_shape_features_1m.parquet"))
    args = ap.parse_args()

    train_lgbm_shape_model(args.shape_path)


if __name__ == "__main__":
    main()
