"""
5-Seed Bagging Ensemble LightGBM with Custom Asymmetric Loss & Precision Tuning.
"""
import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def add_technical_indicators(df: pl.DataFrame) -> pl.DataFrame:
    low_14 = pl.col("low").rolling_min(window_size=14)
    high_14 = pl.col("high").rolling_max(window_size=14)
    stoch_k = ((pl.col("close") - low_14) / (high_14 - low_14 + 1e-8)) * 100
    stoch_d = stoch_k.rolling_mean(window_size=3)

    diff = pl.col("close") - pl.col("close").shift(1)
    gain = pl.when(diff > 0).then(diff).otherwise(0.0).rolling_mean(window_size=14)
    loss = pl.when(diff < 0).then(-diff).otherwise(0.0).rolling_mean(window_size=14)
    rs = gain / (loss + 1e-8)
    rsi_14 = 100 - (100 / (1 + rs))

    bb_mid = pl.col("close").rolling_mean(window_size=20)
    bb_std = pl.col("close").rolling_std(window_size=20)
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_pct = (pl.col("close") - bb_lower) / (bb_upper - bb_lower + 1e-8)

    tr = pl.max_horizontal([
        pl.col("high") - pl.col("low"),
        (pl.col("high") - pl.col("close").shift(1)).abs(),
        (pl.col("low") - pl.col("close").shift(1)).abs()
    ])
    atr_14 = tr.rolling_mean(window_size=14) / pl.col("close")

    return df.with_columns([
        stoch_k.alias("stoch_k"),
        stoch_d.alias("stoch_d"),
        rsi_14.alias("rsi_14"),
        bb_pct.alias("bb_pct"),
        atr_14.alias("atr_14"),
        ((pl.col("close") - pl.col("open")) / pl.col("open")).alias("candle_body"),
        ((pl.col("high") - pl.col("close")) / pl.col("close")).alias("upper_shadow"),
        ((pl.col("close") - pl.col("low")) / pl.col("close")).alias("lower_shadow"),
    ])


def precision_focused_loss(preds, train_data):
    labels = train_data.get_label()
    p = 1.0 / (1.0 + np.exp(-preds))
    w_fp = 5.0
    grad = p * (1 + w_fp) - labels - w_fp * p * labels
    hess = p * (1 - p) * (1 + w_fp * (1 - labels))
    return grad, hess


def train_5seed_ensemble(master_parquet_path: str, target_horizon: int = 5, test_ratio: float = 0.2):
    df_master = pl.read_parquet(master_parquet_path).sort("close_time")
    df_ta = add_technical_indicators(df_master)

    windows = [1, 3, 5, 15, 30, 60]
    exprs = []
    ta_features = ["stoch_k", "stoch_d", "rsi_14", "bb_pct", "atr_14", "candle_body", "upper_shadow", "lower_shadow"]

    for w in windows:
        c_ret = f"ret_{w}m"
        exprs.append(((pl.col("close") - pl.col("close").shift(w)) / (pl.col("close").shift(w) + 1e-8)).alias(c_ret))
        ta_features.append(c_ret)

        c_vir = f"vir_{w}m"
        r_buy = pl.col("um_buy_vol").rolling_sum(window_size=w)
        r_sell = pl.col("um_sell_vol").rolling_sum(window_size=w)
        exprs.append(((r_buy - r_sell) / (r_buy + r_sell + 1e-8)).alias(c_vir))
        ta_features.append(c_vir)

    future_close = pl.col("close").shift(-target_horizon)
    target_expr = pl.when(future_close > pl.col("close")).then(1).otherwise(0).alias("target")

    df_feat = df_ta.with_columns(exprs + [target_expr]).slice(35, len(df_ta) - 40)
    df_clean = df_feat.with_columns([pl.col(c).fill_null(0.0).fill_nan(0.0) for c in ta_features]).filter(pl.col("target").is_not_null())

    split_idx = int(len(df_clean) * (1 - test_ratio))
    df_train = df_clean.slice(0, split_idx - target_horizon)
    df_test = df_clean.slice(split_idx, len(df_clean) - split_idx)

    X_tr, y_tr = df_train.select(ta_features).to_numpy(), df_train["target"].to_numpy()
    X_te, y_te = df_test.select(ta_features).to_numpy(), df_test["target"].to_numpy()

    seeds = [42, 100, 2024, 777, 999]
    train_preds_list = []
    test_preds_list = []

    print(f"Training 5-Seed Ensemble LightGBM ({len(seeds)} Seeds: {seeds})...")

    for seed in seeds:
        train_data = lgb.Dataset(X_tr, label=y_tr)
        valid_data = lgb.Dataset(X_te, label=y_te, reference=train_data)

        params = {
            "objective": precision_focused_loss,
            "metric": "binary_logloss",
            "learning_rate": 0.015,
            "num_leaves": 12,
            "max_depth": 4,
            "min_child_samples": 300,
            "feature_fraction": 0.7,
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

    # Ensemble Averaging
    preds_tr_prob = np.mean(train_preds_list, axis=0)
    preds_te_prob = np.mean(test_preds_list, axis=0)

    # Individual model std variance across 5 seeds
    seed_std = np.std(test_preds_list, axis=0).mean()
    print(f"\n5-Seed Average Out-Of-Sample AUC: {roc_auc_score(y_te, preds_te_prob):.4f}")
    print(f"Average Inter-Seed Prediction Standard Deviation: {seed_std:.6f} (Variance Reduced!)")

    test_days = (df_test["close_time"].max() - df_test["close_time"].min()) / (1000 * 3600 * 24)
    test_close_times = df_test["close_time"].to_numpy()

    print("\n--- 5-Seed Ensemble High Precision Evaluation ---")
    for q in [0.95, 0.98, 0.99, 0.995, 0.998, 0.999]:
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
    ap = argparse.ArgumentParser(description="5-Seed Ensemble LightGBM Trainer")
    ap.add_argument("--master-path", default=str(DATA / "BTCUSDT_master_features_1m.parquet"))
    args = ap.parse_args()

    train_5seed_ensemble(args.master_path)


if __name__ == "__main__":
    main()
