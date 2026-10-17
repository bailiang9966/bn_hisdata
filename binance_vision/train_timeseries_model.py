"""
PyTorch Deep Learning Time-Series Model (Fast CPU GRU) for 5-Minute Event Contract Prediction.
"""
import argparse
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


class TimeSeriesDataset(Dataset):
    """3D Sequence Dataset for PyTorch (N_samples, Seq_Len, N_features)."""
    def __init__(self, features: np.ndarray, targets: np.ndarray, seq_len: int = 15, step: int = 2):
        self.seq_len = seq_len

        # Subsample indices by step to speed up sequence dataset creation
        indices = np.arange(seq_len, len(targets), step)
        self.targets = targets[indices]

        n_samples = len(indices)
        n_features = features.shape[1]

        self.sequences = np.zeros((n_samples, seq_len, n_features), dtype=np.float32)
        for idx_i, idx_t in enumerate(indices):
            self.sequences[idx_i] = features[idx_t - seq_len : idx_t]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        x = torch.tensor(self.sequences[idx], dtype=torch.float32)
        y = torch.tensor(self.targets[idx], dtype=torch.float32)
        return x, y


class FastGRUClassifier(nn.Module):
    """Fast GRU Neural Network for High-Frequency Direction Prediction."""
    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.SiLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        out, _ = self.gru(x)  # (batch, seq_len, hidden_dim)
        last_out = out[:, -1, :]  # (batch, hidden_dim)
        logits = self.fc(last_out).squeeze(-1)
        return logits


def prepare_features(master_parquet_path: str, target_horizon: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    df = pl.read_parquet(master_parquet_path).sort("close_time")

    windows = [1, 3, 5, 15, 30, 60, 240]
    exprs = []
    feature_cols = []

    for w in windows:
        col_ret = f"ret_{w}m"
        exprs.append(((pl.col("close") - pl.col("close").shift(w)) / (pl.col("close").shift(w) + 1e-8)).alias(col_ret))
        feature_cols.append(col_ret)

        col_vir = f"um_vir_{w}m"
        r_buy = pl.col("um_buy_vol").rolling_sum(window_size=w)
        r_sell = pl.col("um_sell_vol").rolling_sum(window_size=w)
        exprs.append(((r_buy - r_sell) / (r_buy + r_sell + 1e-8)).alias(col_vir))
        feature_cols.append(col_vir)

        col_basis = f"basis_change_{w}m"
        exprs.append((pl.col("basis") - pl.col("basis").shift(w)).alias(col_basis))
        feature_cols.append(col_basis)

        if w in [5, 15, 30, 60, 240]:
            col_z = f"price_robust_zscore_{w}m"
            r_med = pl.col("close").rolling_median(window_size=w)
            r_mad = (pl.col("close") - r_med).abs().rolling_median(window_size=w)
            exprs.append(((pl.col("close") - r_med) / (1.4826 * r_mad + 1e-8)).alias(col_z))
            feature_cols.append(col_z)

    for w in [5, 15, 60]:
        if "sum_open_interest" in df.columns:
            col_oi = f"oi_change_{w}m"
            exprs.append(((pl.col("sum_open_interest") - pl.col("sum_open_interest").shift(w)) /
                          (pl.col("sum_open_interest").shift(w) + 1e-8)).alias(col_oi))
            feature_cols.append(col_oi)

    base_features = [
        "basis", "basis_rate", "um_buy_vol", "um_sell_vol", "um_count",
        "spot_buy_vol", "spot_sell_vol", "spot_count",
        "sum_open_interest", "count_toptrader_long_short_ratio",
        "sum_toptrader_long_short_ratio", "count_long_short_ratio"
    ]
    for bf in base_features:
        if bf in df.columns and bf not in feature_cols:
            feature_cols.append(bf)

    future_close = pl.col("close").shift(-target_horizon)
    target_expr = pl.when(future_close > pl.col("close")).then(1.0).otherwise(0.0).alias("target")

    df_featured = df.with_columns(exprs + [target_expr])
    max_w = max(windows)
    df_clean = df_featured.slice(max_w, len(df_featured) - max_w - target_horizon)

    clean_exprs = [pl.col(c).fill_null(0.0).fill_nan(0.0).clip(-1e5, 1e5).alias(c) for c in feature_cols]
    df_clean = df_clean.with_columns(clean_exprs).filter(pl.col("target").is_not_null())

    X = df_clean.select(feature_cols).to_numpy()
    y = df_clean["target"].to_numpy()
    times = df_clean["close_time"].to_numpy()

    return X, y, times, feature_cols


def train_pytorch_timeseries_model(
    master_parquet_path: str,
    seq_len: int = 15,
    target_horizon: int = 5,
    epochs: int = 3,
    batch_size: int = 2048,
):
    print(f"Loading master table from {master_parquet_path}...")
    X, y, times, feature_cols = prepare_features(master_parquet_path, target_horizon)

    split_idx = int(len(X) * 0.8)

    mean_tr = np.mean(X[:split_idx], axis=0)
    std_tr = np.std(X[:split_idx], axis=0) + 1e-8
    X_scaled = (X - mean_tr) / std_tr

    X_train_arr, y_train_arr = X_scaled[:split_idx], y[:split_idx]
    X_test_arr, y_test_arr = X_scaled[split_idx:], y[split_idx:]

    indices_test = np.arange(seq_len, len(y_test_arr), 2)
    times_test = times[split_idx:][indices_test]

    ds_train = TimeSeriesDataset(X_train_arr, y_train_arr, seq_len=seq_len, step=2)
    ds_test = TimeSeriesDataset(X_test_arr, y_test_arr, seq_len=seq_len, step=2)

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Fast GRU Time-Series model on device: {device}")

    model = FastGRUClassifier(input_dim=len(feature_cols), hidden_dim=32).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for bx, by in dl_train:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(by)

        train_loss /= len(ds_train)

        model.eval()
        test_preds = []
        with torch.no_grad():
            for bx, by in dl_test:
                bx = bx.to(device)
                logits = model(bx)
                probs = torch.sigmoid(logits).cpu().numpy()
                test_preds.extend(probs)

        test_preds = np.array(test_preds)
        y_test_eval = ds_test.targets
        auc = roc_auc_score(y_test_eval, test_preds)
        print(f"Epoch {epoch:2d}/{epochs:2d} | Train Loss: {train_loss:.4f} | Out-Of-Sample AUC: {auc:.4f}")

    # Final Evaluation & De-overlapped High-Confidence Filter
    model.eval()
    test_preds = []
    with torch.no_grad():
        for bx, by in dl_test:
            bx = bx.to(device)
            logits = model(bx)
            probs = torch.sigmoid(logits).cpu().numpy()
            test_preds.extend(probs)

    preds_test = np.array(test_preds)
    y_test_eval = ds_test.targets

    test_days = (times_test[-1] - times_test[0]) / (1000 * 3600 * 24)

    print("\n=== Out-of-Sample High Confidence Precision Curve ===")
    for q in [0.80, 0.90, 0.95, 0.98, 0.99]:
        thresh_dev = np.quantile(np.abs(preds_test - 0.5), q)
        long_thresh = 0.5 + thresh_dev
        short_thresh = 0.5 - thresh_dev

        selected_trades = []
        last_t = -999999
        for i in range(len(preds_test)):
            p = preds_test[i]
            t = times_test[i]
            if (p >= long_thresh or p <= short_thresh) and (t - last_t >= target_horizon * 60 * 1000):
                pred_label = 1 if p >= long_thresh else 0
                actual_label = y_test_eval[i]
                selected_trades.append((t, pred_label, actual_label))
                last_t = t

        n_tr = len(selected_trades)
        if n_tr > 0:
            win = sum(1 for tr in selected_trades if tr[1] == tr[2]) / n_tr
            freq = n_tr / test_days
            print(f"Quantile {q*100:4.0f}% | Thresh: Long >= {long_thresh:.4f}, Short <= {short_thresh:.4f} | Trades: {n_tr:4d} | Freq: {freq:5.2f}/day | Win Rate: {win*100:5.2f}%")


def main():
    ap = argparse.ArgumentParser(description="Train PyTorch GRU time-series model")
    ap.add_argument("--master-path", default=str(DATA / "BTCUSDT_master_features_1m.parquet"))
    ap.add_argument("--seq-len", type=int, default=15, help="Sequence window length in minutes (default: 15)")
    ap.add_argument("--epochs", type=int, default=3, help="Number of training epochs (default: 3)")
    args = ap.parse_args()

    train_pytorch_timeseries_model(args.master_path, seq_len=args.seq_len, epochs=args.epochs)


if __name__ == "__main__":
    main()
