# rebuild_all_covs.py
import os, glob, numpy as np, pandas as pd, argparse

def main(features_dir, all_csv, out_cov):
    df = pd.read_csv(all_csv)
    subjects = list(df['subject'].astype(str).unique())
    cov_list = []
    index = []
    cur = 0
    for s in subjects:
        fn = os.path.join(features_dir, f"covs_{s}.npy")
        if not os.path.exists(fn):
            print("Missing cov file for", s, "-> skipping")
            continue
        arr = np.load(fn, allow_pickle=True)
        # ensure float32
        arr = arr.astype('float32')
        n = arr.shape[0]
        cov_list.append(arr)
        index.append((s, cur, cur + n))
        cur += n
        print("Loaded", fn, "shape", arr.shape)
    if not cov_list:
        raise ValueError("No cov files loaded")
    all_covs = np.concatenate(cov_list, axis=0)  # (total_epochs, n_bands, n_ch, n_ch)
    np.save(out_cov, all_covs)
    np.save(os.path.join(features_dir, "cov_index.npy"), index)
    print("Saved all_covs:", out_cov, "shape:", all_covs.shape)
    print("Saved cov_index.npy to", features_dir)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--features_dir", default=r"D:\Data_cog\features")
    p.add_argument("--all_csv", default=r"D:\Data_cog\features\all_features_clean.csv")
    p.add_argument("--out", default=r"D:\Data_cog\features\all_covs.npy")
    args = p.parse_args()
    main(args.features_dir, args.all_csv, args.out)
