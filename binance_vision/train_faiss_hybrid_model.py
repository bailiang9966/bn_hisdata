"""
FAISS Pattern Similarity + Microstructure Hybrid Model Trainer.
Combines K-NN Historical Waveform Memory (faiss_hist_win_rate) with Microstructure Features.
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


def train_faiss_hybrid_model(master_path: str, faiss_path: str, target_horizon: int = 5, test_ratio: float = 0.2):
    df_master = pl.read_parquet(master_path).sort("close_time")
    df_faiss = pl.read_parquet(faiss_path).sort("close_time")

    # Join FAISS pattern features
    df_combined = df_master.join(df_faiss, on="close_time", how="inner")

    # Add Technical Indicators
    low_14 = pl.col("low").rolling_min(window_size=14)
    high_14 = pl.col("high").rolling_max(window_size=14)
    stoch_k = ((pl.col("close") - low_14) / (high_14 - low_14 + 1e-8)) * 100.0
    stoch_d = stoch_k.rolling_mean(window_size=3)

    diff = pl.col("close") - pl.col("close").shift(1)
    gain = pl.when(diff > 0).then(diff).otherwise(0.0).rolling_mean(window_size=14)
    loss = pl.when(diff < 0).then(-diff).otherwise(0.0).rolling_mean(window_size=14)
    rsi_14 = 100.0 - (100.0 / (1.0 + (gain / (loss + 1e-8))))

    df_ta = df_combined.with_columns([
        stoch_k.alias("stoch_k"),
        stoch_d.alias("stoch_d"),
        rsi_14.alias("rsi_14"),
        ((pl.col("close") - pl.col("open")) / pl.col("open")).alias("candle_body"),
        ((pl.col("high") - pl.col("close")) / pl.col("close")).alias("upper_shadow"),
        ((pl.col("close") - pl.col("low")) / pl.col("close")).alias("lower_shadow"),
    ])

    windows = [1, 3, 5, 15, 30, 60]
    exprs = []
    feature_cols = ["faiss_hist_win_rate", "faiss_min_dist", "stoch_k", "stoch_d", "rsi_14", "candle_body", "upper_shadow", "lower_shadow"]

    for w in windows:
        c_ret = f"ret_{w}m"
        exprs.append(((pl.col("close") - pl.col("close").shift(w)) / (pl.col("close").shift(w) + 1e-8)).alias(c_ret))
        feature_cols.append(c_ret)

        c_vir = f"vir_{w}m"
        r_buy = pl.col("um_buy_vol").rolling_sum(window_size=w)
        r_sell = pl.col("um_sell_vol").rolling_sum(window_size=w)
        exprs.append(((r_buy - r_sell) / (r_buy + r_sell + 1e-8)).alias(c_vir))
        feature_cols.append(c_vir)

    base_cols = ["basis", "um_buy_vol", "um_sell_vol", "um_count", "sum_open_interest", "sum_toptrader_long_short_ratio"]
    for bc in base_cols:
        if bc in df_ta.columns and bc not in feature_cols:
            feature_cols.append(bc)

    future_close = pl.col("close").shift(-target_horizon)
    target_expr = pl.when(future_close > pl.col("close")).then(1).otherwise(0).alias("target")

    df_feat = df_ta.with_columns(exprs + [target_expr]).slice(35, len(df_ta) - 40)
    df_clean = df_feat.with_columns([pl.col(c).fill_null(0.0).fill_nan(0.0) for c in feature_cols]).filter(pl.col("target").is_not_null())

    split_idx = int(len(df_clean) * (1 - test_ratio))
    df_train = df_clean.slice(0, split_idx - target_horizon)
    df_test = df_clean.slice(split_idx, len(df_clean) - split_idx)

    X_tr, y_tr = df_train.select(feature_cols).to_numpy(), df_train["target"].to_numpy()
    X_te, y_te = df_test.select(feature_cols).to_numpy(), df_test["target"].to_numpy()

    seeds = [42, 100, 2024, 777, 999]
    test_preds_list = []
    train_preds_list = []

    print(f"Training FAISS Hybrid 5-Seed Ensemble LightGBM on {len(feature_cols)} features...")

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
    print(f"\nFAISS Hybrid Model Out-Of-Sample AUC: {auc:.4f}")

    test_days = (df_test["close_time"].max() - df_test["close_time"].min()) / (1000 * 3600 * 24)
    test_close_times = df_test["close_time"].to_numpy()

    print("\n--- FAISS Pattern + Microstructure Hybrid Model Results ---")
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
    ap = argparse.ArgumentParser(description="Train FAISS Hybrid Model")
    ap.add_argument("--master-path", default=str(DATA / "BTCUSDT_master_features_1m.parquet"))
    ap.add_argument("--faiss-path", default=str(DATA / "BTCUSDT_faiss_features_1m.parquet"))
    args = ap.parse_args()

    train_faiss_hybrid_model(args.master_path, args.faiss_path)


if __name__ == "__main__":
    main()
