import os
import pandas as pd

import sys
sys.path.append(os.path.abspath("src"))

from data.load_data import load_data
from data.preprocess import preprocess_data
from features.build_features import build_features

DATA_PATH = "data/external/Telco-Customer-Churn.csv"  # adjust to your file path
TARGET_COL = "Churn"


def main():
    print("=== Testing Phase 1: Load → Preprocess → Build Features ===")

    print("Loading data...")
    df = load_data(DATA_PATH)
    print(f"Data loaded. Shape: {df.shape}")
    print(df.head(3))

    print("Preprocessing data...")
    df_clean = preprocess_data(df, target_col=TARGET_COL)
    print(f"Data after preprocessing. Shape: {df_clean.shape}")
    print(df_clean.head(3))

    print("Building features...")
    df_features = build_features(df_clean, target_col=TARGET_COL)
    print(f"Data after feature engineering. Shape: {df_features.shape}")
    print(df_features.head(3))

    print("Phase 1 pipeline completed successfully!")

if __name__ == "__main__":
    main()