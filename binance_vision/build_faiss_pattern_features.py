"""
FAISS Vector Similarity Pattern Feature Generator (Fast IVFFlat Indexing).
"""
import argparse
from pathlib import Path

import faiss
import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def compute_stoch_k(close: np.ndarray, low: np.ndarray, high: np.ndarray, n: int = 80) -> np.ndarray:
    df = pl.DataFrame({"close": close, "low": low, "high": high})
    low_n = pl.col("low").rolling_min(window_size=n)
    high_n = pl.col("high").rolling_max(window_size=n)
    stoch_k = ((pl.col("close") - low_n) / (high_n - low_n + 1e-8)) * 100.0
    return df.with_columns(stoch_k.alias("stoch_k"))["stoch_k"].to_numpy()


def extract_faiss_pattern_features(
    master_parquet_path: str,
    pattern_len: int = 15,
    top_k: int = 50,
    target_horizon: int = 5,
    test_ratio: float = 0.2,
) -> pl.DataFrame:
    print(f"Loading {master_parquet_path}...")
    df = pl.read_parquet(master_parquet_path).sort("close_time")

    close_arr = df["close"].to_numpy()
    high_arr = df["high"].to_numpy()
    low_arr = df["low"].to_numpy()
    times = df["close_time"].to_numpy()

    stoch_arr = compute_stoch_k(close_arr, low_arr, high_arr, n=80)

    ret_arr = np.zeros(len(close_arr))
    ret_arr[1:] = (close_arr[1:] - close_arr[:-1]) / (close_arr[:-1] + 1e-8)

    future_close = np.roll(close_arr, -target_horizon)
    target_arr = (future_close > close_arr).astype(np.float32)

    start_idx = 80
    end_idx = len(close_arr) - target_horizon
    n_samples = end_idx - start_idx
    feature_dim = pattern_len * 2

    # Vectorized 2D matrix extraction
    shape = (n_samples, pattern_len)
    strides = (stoch_arr.strides[0], stoch_arr.strides[0])

    stoch_mat = np.lib.stride_tricks.as_strided(stoch_arr[start_idx - pattern_len + 1:], shape=shape, strides=strides)
    ret_mat = np.lib.stride_tricks.as_strided(ret_arr[start_idx - pattern_len + 1:], shape=shape, strides=strides)

    stoch_norm = np.nan_to_num((stoch_mat - np.mean(stoch_mat, axis=1, keepdims=True)) / (np.std(stoch_mat, axis=1, keepdims=True) + 1e-8))
    ret_norm = np.nan_to_num((ret_mat - np.mean(ret_mat, axis=1, keepdims=True)) / (np.std(ret_mat, axis=1, keepdims=True) + 1e-8))

    vectors = np.ascontiguousarray(np.hstack([stoch_norm, ret_norm]), dtype=np.float32)
    vectors = np.nan_to_num(vectors, nan=0.0, posinf=0.0, neginf=0.0)
    valid_targets = target_arr[start_idx:end_idx]
    valid_times = times[start_idx:end_idx]

    split_idx = int(n_samples * (1 - test_ratio))

    # Subsample training database every 2 mins for fast indexing
    train_sub_indices = np.arange(0, split_idx, 2)
    X_train_vec = vectors[train_sub_indices]
    y_train_vec = valid_targets[train_sub_indices]

    print(f"Building Fast FAISS IVFFlat Index on {len(X_train_vec)} historical training patterns...")
    nlist = 100
    quantizer = faiss.IndexFlatL2(feature_dim)
    index = faiss.IndexIVFFlat(quantizer, feature_dim, nlist)
    index.train(X_train_vec)
    index.add(X_train_vec)
    index.nprobe = 10

    print(f"FAISS K-NN Search (K={top_k}) for {n_samples} vectors...")
    distances, indices = index.search(vectors, top_k)

    indices_clipped = np.clip(indices, 0, len(y_train_vec) - 1)
    faiss_win_rates = np.mean(y_train_vec[indices_clipped], axis=1)
    faiss_min_dists = distances[:, 0]

    df_out = pl.DataFrame({
        "close_time": valid_times,
        "faiss_hist_win_rate": faiss_win_rates,
        "faiss_min_dist": faiss_min_dists,
    })

    return df_out


def main():
    ap = argparse.ArgumentParser(description="Build FAISS Pattern Features")
    ap.add_argument("--master-path", default=str(DATA / "BTCUSDT_master_features_1m.parquet"))
    ap.add_argument("--output", default=str(DATA / "BTCUSDT_faiss_features_1m.parquet"))
    ap.add_argument("--top-k", type=int, default=50)
    args = ap.parse_args()

    df_faiss = extract_faiss_pattern_features(args.master_path, top_k=args.top_k)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_faiss.write_parquet(out_path, compression="zstd")
    print(f"Successfully generated FAISS pattern feature table: {out_path} ({len(df_faiss)} rows)")


if __name__ == "__main__":
    main()
