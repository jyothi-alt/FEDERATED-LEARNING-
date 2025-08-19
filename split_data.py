# split_data.py
import argparse
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from utils import FEATURES, TARGET
from sklearn.model_selection import train_test_split

def split_and_save_with_scaling(csv_path, out_dir, n_clients=3, seed=42):
    os.makedirs(out_dir, exist_ok=True)

    # Load dataset
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)

   
    le = LabelEncoder()
    df[TARGET] = le.fit_transform(df[TARGET].astype(str))

  
    scaler = StandardScaler()
    df[FEATURES] = scaler.fit_transform(df[FEATURES].astype(float))

  
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

   
    parts = np.array_split(df, n_clients)
    paths = []
    for i, part in enumerate(parts):
        p = os.path.join(out_dir, f"client_{i}.csv")
        part.to_csv(p, index=False)
        paths.append(p)

    print(f" Global scaling applied. Client CSVs created: {paths}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset with global scaling")
    parser.add_argument("--csv", default="data/Maternal-Health-Risk-Data-Set.csv")
    parser.add_argument("--out_dir", default="data/splits")
    parser.add_argument("--n_clients", type=int, default=3)
    args = parser.parse_args()

    split_and_save_with_scaling(args.csv, args.out_dir, args.n_clients)
