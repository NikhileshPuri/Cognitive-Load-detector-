#!/usr/bin/env python3
# D:\Data_cog\Scripts\inspect_covs_shapes.py
import os, glob, numpy as np
from collections import Counter

FEATURES_DIR = r"D:\Data_cog\features"
ALL_COVS = os.path.join(FEATURES_DIR, "all_covs.npy")

def quick():
    print("Looking for per-subject cov files: covs_sub-*.npy")
    cov_files = sorted(glob.glob(os.path.join(FEATURES_DIR, "covs_sub-*.npy")))
    print(f"Found {len(cov_files)} cov files")
    shapes = Counter()
    total_epochs = 0
    for f in cov_files:
        try:
            a = np.load(f, mmap_mode='r')
            shapes[a.shape] += 1
            total_epochs += a.shape[0]
        except Exception as e:
            print("Failed to load", f, e)
    for s,c in shapes.items():
        print(" shape:", s, "count:", c)
    print("Total epochs across subjects (sum first dim):", total_epochs)
    if os.path.exists(ALL_COVS):
        try:
            allc = np.load(ALL_COVS, mmap_mode='r')
            print("Loaded all_covs.npy shape:", allc.shape, "dtype:", allc.dtype)
        except Exception as e:
            print("Failed to load all_covs.npy:", e)
    else:
        print("No all_covs.npy found at", ALL_COVS)

if __name__ == "__main__":
    quick()
