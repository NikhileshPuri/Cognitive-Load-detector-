"""
Split the aggregated ML-ready dataset into train/test (and optionally validation) sets.
Stratified by label_enc to preserve class balance.

Usage:
  python prepare_train_test_split.py --data "D:/Data_cog/features/all_features_clean.csv"
"""

import os, argparse, pandas as pd
from sklearn.model_selection import train_test_split

def prepare_splits(data_csv, out_dir, test_size=0.2, val_size=0.1, random_state=42):
    df = pd.read_csv(data_csv)
    os.makedirs(out_dir, exist_ok=True)

    label_col = None
    for c in df.columns:
        if c.lower().endswith('_enc'):
            label_col = c
            break
    if not label_col:
        raise ValueError("No encoded label column found (expected *_enc).")

    print(f"✅ Loaded {len(df)} samples from {data_csv}")
    print(f"Stratifying splits by '{label_col}'...")

    # Split train/test
    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df[label_col], random_state=random_state
    )

    # Optional validation split from train
    if val_size > 0:
        val_rel_size = val_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_df, test_size=val_rel_size, stratify=train_df[label_col], random_state=random_state
        )
    else:
        val_df = None

    # Save outputs
    train_path = os.path.join(out_dir, "train.csv")
    test_path = os.path.join(out_dir, "test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"✅ Train: {len(train_df)} | Test: {len(test_df)}")

    if val_df is not None:
        val_path = os.path.join(out_dir, "val.csv")
        val_df.to_csv(val_path, index=False)
        print(f"✅ Validation: {len(val_df)}")
    else:
        val_path = None

    return train_path, test_path, val_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=r"D:\Data_cog\features\all_features_clean.csv")
    p.add_argument("--out_dir", default=r"D:\Data_cog\features\splits")
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--val_size", type=float, default=0.1)
    p.add_argument("--random_state", type=int, default=42)
    args = p.parse_args()

    prepare_splits(args.data, args.out_dir, args.test_size, args.val_size, args.random_state)
