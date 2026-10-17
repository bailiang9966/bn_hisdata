"""
PyTorch Deep 1D-ResNet Pattern Recognition Network on Long-Period Oscillator Waveform Tensors.
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


class ResNetBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        res = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += res
        return self.relu(out)


class ResNet1DClassifier(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int = 32):
        super().__init__()
        self.prep = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer1 = ResNetBlock1D(hidden_dim, hidden_dim * 2)
        self.layer2 = ResNetBlock1D(hidden_dim * 2, hidden_dim * 4)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 4, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x shape: (batch_size, in_channels, seq_len)
        out = self.prep(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.pool(out).squeeze(-1)
        return self.fc(out).squeeze(-1)


class ShapeSequenceDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray, seq_len: int = 20, step: int = 2):
        indices = np.arange(seq_len, len(targets), step)
        self.targets = targets[indices]

        n_samples = len(indices)
        n_features = features.shape[1]

        # Shape tensor: (n_samples, n_features, seq_len)
        self.sequences = np.zeros((n_samples, n_features, seq_len), dtype=np.float32)
        for idx_i, idx_t in enumerate(indices):
            # Transpose so channels is dim 1: (n_features, seq_len)
            self.sequences[idx_i] = features[idx_t - seq_len : idx_t].T

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        x = torch.tensor(self.sequences[idx], dtype=torch.float32)
        y = torch.tensor(self.targets[idx], dtype=torch.float32)
        return x, y


def train_resnet_shape_model(shape_parquet_path: str, epochs: int = 3, batch_size: int = 1024):
    df_shape = pl.read_parquet(shape_parquet_path).sort("close_time")

    # Select long-period bounded oscillator shape columns
    shape_cols = [c for c in df_shape.columns if ("stoch" in c or "rsi" in c or "confluence" in c or "shadow" in c or "body" in c) and c != "target"]

    # Filter nulls & target
    clean_exprs = [pl.col(c).fill_null(0.0).fill_nan(0.0).clip(-100.0, 100.0).alias(c) for c in shape_cols]
    df_clean = df_shape.with_columns(clean_exprs).filter(pl.col("target").is_not_null())

    X = df_clean.select(shape_cols).to_numpy()
    y = df_clean["target"].to_numpy()
    times = df_clean["close_time"].to_numpy()

    split_idx = int(len(X) * 0.8)

    # Standardize
    mean_tr = np.mean(X[:split_idx], axis=0)
    std_tr = np.std(X[:split_idx], axis=0) + 1e-8
    X_scaled = (X - mean_tr) / std_tr

    seq_len = 20
    indices_test = np.arange(seq_len, len(y[split_idx:]), 2)
    times_test = times[split_idx:][indices_test]

    ds_train = ShapeSequenceDataset(X_scaled[:split_idx], y[:split_idx], seq_len=seq_len, step=2)
    ds_test = ShapeSequenceDataset(X_scaled[split_idx:], y[split_idx:], seq_len=seq_len, step=2)

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training 1D-ResNet Pattern Classifier on {len(shape_cols)} shape features...")

    model = ResNet1DClassifier(in_channels=len(shape_cols), hidden_dim=32).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        for bx, by in dl_train:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * len(by)

        tr_loss /= len(ds_train)

        model.eval()
        test_preds = []
        with torch.no_grad():
            for bx, by in dl_test:
                bx = bx.to(device)
                logits = model(bx)
                probs = torch.sigmoid(logits).cpu().numpy()
                test_preds.extend(probs)

        test_preds = np.array(test_preds)
        auc = roc_auc_score(ds_test.targets, test_preds)
        print(f"Epoch {epoch}/{epochs} | Train Loss: {tr_loss:.4f} | Out-Of-Sample AUC: {auc:.4f}")

    # High Confidence Evaluation Curve
    test_days = (times_test[-1] - times_test[0]) / (1000 * 3600 * 24)
    y_test_eval = ds_test.targets

    print("\n=== 1D-ResNet Waveform Pattern Precision Curve ===")
    for q in [0.80, 0.90, 0.95, 0.98, 0.99]:
        thresh_dev = np.quantile(np.abs(test_preds - 0.5), q)
        long_thresh = 0.5 + thresh_dev
        short_thresh = 0.5 - thresh_dev

        selected_trades = []
        last_t = -999999
        for i in range(len(test_preds)):
            p = test_preds[i]
            t = times_test[i]
            if (p >= long_thresh or p <= short_thresh) and (t - last_t >= 5 * 60 * 1000):
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
    ap = argparse.ArgumentParser(description="Train 1D-ResNet Shape Pattern Neural Network")
    ap.add_argument("--shape-path", default=str(DATA / "BTCUSDT_shape_features_1m.parquet"))
    ap.add_argument("--epochs", type=int, default=3)
    args = ap.parse_args()

    train_resnet_shape_model(args.shape_path, epochs=args.epochs)


if __name__ == "__main__":
    main()
