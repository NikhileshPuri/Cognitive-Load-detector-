# aggregate_features.py
import os, glob, argparse, numpy as np, pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def aggregate(features_dir, out_raw, out_clean):
    csv_files = sorted(glob.glob(os.path.join(features_dir, "sub-*_features.csv")))
    if not csv_files:
        raise FileNotFoundError("No *_features.csv found in " + features_dir)
    dfs=[]
    for f in csv_files:
        df = pd.read_csv(f)
        subj = os.path.basename(f).split("_")[0]
        if 'subject' not in df.columns:
            df['subject'] = subj
        if 'epoch_idx' not in df.columns:
            df['epoch_idx'] = range(len(df))
        if 'task' not in df.columns and 'label' in df.columns:
            df['task'] = df['label']
        df['uid'] = df['subject'].astype(str) + "_" + df['task'].astype(str) + "_" + df['epoch_idx'].astype(str)
        dfs.append(df)
    all_df = pd.concat(dfs, ignore_index=True)
    all_df.to_csv(out_raw, index=False)
    # encode label/task
    label_col = 'task' if 'task' in all_df.columns else ('label' if 'label' in all_df.columns else None)
    if label_col is None:
        raise ValueError("No label/task column found in aggregated features.")
    le = LabelEncoder()
    all_df[label_col + "_enc"] = le.fit_transform(all_df[label_col].astype(str))
    # normalize numeric columns except metadata
    exclude = ['subject','session','task','label','uid','epoch_idx', label_col + "_enc"]
    num_cols = [c for c in all_df.columns if c not in exclude and np.issubdtype(all_df[c].dtype, np.number)]
    scaler = StandardScaler()
    all_df[num_cols] = scaler.fit_transform(all_df[num_cols].values)
    all_df.to_csv(out_clean, index=False)
    print("Saved raw:", out_raw)
    print("Saved clean:", out_clean)
    print("Classes:", list(le.classes_))
    return all_df, num_cols

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--features_dir', default=r"D:\Data_cog\features")
    p.add_argument('--out_raw', default=r"D:\Data_cog\features\all_features_raw.csv")
    p.add_argument('--out_clean', default=r"D:\Data_cog\features\all_features_clean.csv")
    args = p.parse_args()
    aggregate(args.features_dir, args.out_raw, args.out_clean)

