# utils.py
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split

TARGET = "RiskLevel"
FEATURES = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]

def load_full_csv(csv_path: str):
    """Load CSV, basic cleaning, returns pandas.DataFrame"""
    df = pd.read_csv(csv_path)
    # Keep only expected columns if present
    keep = [c for c in FEATURES + [TARGET] if c in df.columns]
    df = df[keep].dropna(subset=[TARGET] + FEATURES).reset_index(drop=True)
    return df

def encode_labels(df):
    """Encode target column to integer labels and return encoder (LabelEncoder)"""
    le = LabelEncoder()
    df[TARGET] = le.fit_transform(df[TARGET].astype(str))
    return df, le

def split_and_save(csv_path, out_dir, n_clients=3, seed=42):
    """
    Shuffle data and split into n_clients CSV files saved as out_dir/client_{i}.csv
    """
    os.makedirs(out_dir, exist_ok=True)
    df = load_full_csv(csv_path)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    parts = np.array_split(df, n_clients)
    paths = []
    for i, part in enumerate(parts):
        p = os.path.join(out_dir, f"client_{i}.csv")
        part.to_csv(p, index=False)
        paths.append(p)
    return paths

def get_partition_from_df(df, client_id: int, n_clients: int, seed=42):
    """Return train/test arrays for a given client partition index"""
    # stratified-ish partition to keep label distribution if possible
    if n_clients <= 1:
        df_part = df.copy()
    else:
        # Stratified split into n_clients using StratifiedKFold
        try:
            X = df[FEATURES].values
            y = df[TARGET].values
            skf = StratifiedKFold(n_splits=n_clients, shuffle=True, random_state=seed)
            parts = []
            for _, idx in skf.split(X, y):
                parts.append(df.iloc[idx])
            df_part = parts[client_id]
        except Exception:
            # fallback to simple chunking
            order = np.arange(len(df))
            rng = np.random.RandomState(seed)
            rng.shuffle(order)
            chunks = np.array_split(order, n_clients)
            df_part = df.iloc[chunks[client_id]]
    # local train/val split
    if len(df_part) == 0:
        return None, None, None, None
    train_df, val_df = train_test_split(df_part, test_size=0.2, random_state=seed, stratify=df_part[TARGET] if len(df_part[TARGET].unique())>1 else None)
    Xtr = train_df[FEATURES].astype(float).values
    ytr = train_df[TARGET].astype(int).values
    Xval = val_df[FEATURES].astype(float).values
    yval = val_df[TARGET].astype(int).values
    return Xtr, ytr, Xval, yval

def load_client_data_mode(base_csv=None, client_csv=None, client_id=0, n_clients=5, seed=42):
    """
    Two modes:
     - If client_csv provided -> load that file (per-client CSVs)
     - Else -> load base_csv and partition by client_id among n_clients
    Returns: X_train, y_train, X_val, y_val
    """
    if client_csv:
        df = load_full_csv(client_csv)
        df, _ = encode_labels(df)
        # local train/val
        X = df[FEATURES].astype(float).values
        y = df[TARGET].astype(int).values
        if len(y) == 0:
            return None, None, None, None
        Xtr, Xval, ytr, yval = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y if len(np.unique(y))>1 else None)
        return Xtr, ytr, Xval, yval
    else:
        df = load_full_csv(base_csv)
        df, _ = encode_labels(df)
        return get_partition_from_df(df, client_id, n_clients, seed=seed)
