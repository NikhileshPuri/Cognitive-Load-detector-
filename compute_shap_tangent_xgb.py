"""
compute_shap_tangent_xgb.py – FIXED for multiclass XGBoost (handles 3D array output)

Load a pre‑trained Tangent Space + XGBoost pipeline and compute SHAP values.
Works with common output shapes of shap.TreeExplainer for multiclass:
- list of arrays (one per class), each shape (n_samples, n_features)
- single 3D array of shape (n_samples, n_features, n_classes)
"""

import os
import numpy as np
import pandas as pd
import joblib
import shap
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ========== CONFIGURATION (EDIT THESE PATHS) ==========
MODEL_PATH = "D:/Data_cog/models/tangent_xgb.joblib"
ALL_CSV = "D:/Data_cog/features/all_features_clean.csv"
ALL_COVS = "D:/Data_cog/features/all_covs.npy"
TRAIN_CSV = "D:/Data_cog/features/splits/train.csv"
SHAP_SAMPLE_SIZE = 500          # Number of training samples to use for SHAP (set to None for all)
OUT_PREFIX = "D:/Data_cog/models/tangent_xgb_shap"   # Output prefix for SHAP files
RANDOM_SEED = 42
# =====================================================

def regularize_covs(covs, reg):
    """Add reg * I to each covariance matrix (per band)."""
    if reg == 0:
        return covs
    _, n_bands, n_ch, _ = covs.shape
    covs_reg = covs.copy()
    I = np.eye(n_ch, dtype=covs.dtype) * reg
    for b in range(n_bands):
        covs_reg[:, b, :, :] += I
    return covs_reg

def main():
    print("1. Loading pipeline...")
    pipeline = joblib.load(MODEL_PATH)
    ts_list = pipeline['ts_list']
    scaler = pipeline.get('scaler', None)
    clf = pipeline['classifier']   # xgboost.Booster
    reg = pipeline['reg']
    band_names = pipeline.get('band_names', ['theta', 'alpha', 'beta'])

    print("2. Loading data indices...")
    df_all = pd.read_csv(ALL_CSV)
    train_df = pd.read_csv(TRAIN_CSV)
    uid_to_idx = {uid: i for i, uid in enumerate(df_all['uid'].astype(str))}
    train_idx = np.array([uid_to_idx[uid] for uid in train_df['uid'].astype(str) if uid in uid_to_idx], dtype=int)
    print(f"   Found {len(train_idx)} training samples out of {len(df_all)} total.")

    print("3. Loading covariance matrices (memory‑mapped)...")
    covs = np.load(ALL_COVS, mmap_mode='r')

    print("4. Regularizing covariances...")
    covs_reg = regularize_covs(covs, reg)

    print("5. Transforming to tangent space using pre‑fitted TS objects...")
    X_bands = []
    for b, ts in enumerate(ts_list):
        Xb = ts.transform(covs_reg[:, b, :, :])   # (n_epochs, feat_b)
        X_bands.append(Xb)
        print(f"   Band {b} ({band_names[b]}) transformed -> {Xb.shape}")
    X_all = np.concatenate(X_bands, axis=1)
    print(f"   Total tangent feature dimension: {X_all.shape[1]}")

    if scaler is not None:
        print("6. Applying scaler...")
        X_all = scaler.transform(X_all)

    # Select training samples
    X_train = X_all[train_idx]
    print(f"7. Training samples available: {X_train.shape[0]}")

    # Subsample for SHAP (speed + memory)
    if SHAP_SAMPLE_SIZE and X_train.shape[0] > SHAP_SAMPLE_SIZE:
        rng = np.random.default_rng(RANDOM_SEED)
        sample_idx = rng.choice(X_train.shape[0], size=SHAP_SAMPLE_SIZE, replace=False)
        X_sample = X_train[sample_idx]
        print(f"   Using {SHAP_SAMPLE_SIZE} samples for SHAP.")
    else:
        X_sample = X_train
        print(f"   Using all {X_sample.shape[0]} training samples for SHAP.")

    print("8. Computing SHAP values (this may take a while)...")
    explainer = shap.TreeExplainer(clf)
    shap_vals = explainer.shap_values(X_sample)   # can be list or 3D array
    print(f"   Type of shap_vals: {type(shap_vals)}")
    if isinstance(shap_vals, list):
        print(f"   List length: {len(shap_vals)}  (should be number of classes)")
        # Assume each element is (n_samples, n_features)
        # Stack into (n_classes, n_samples, n_features)
        shap_array = np.stack(shap_vals, axis=0)
        n_classes = shap_array.shape[0]
        n_samples = shap_array.shape[1]
        n_features = shap_array.shape[2]
        print(f"   After stacking: shape = ({n_classes}, {n_samples}, {n_features})")
    elif hasattr(shap_vals, 'shape'):
        shap_array = shap_vals
        print(f"   Shape of shap_vals: {shap_array.shape}")
        # Expected shape: (n_samples, n_features, n_classes) or (n_samples, n_classes, n_features)
        # We want (n_classes, n_samples, n_features) for uniform processing
        if shap_array.ndim == 3:
            # Try to infer orientation
            if shap_array.shape[-1] > 1 and shap_array.shape[1] > 1:
                # Assume (n_samples, n_features, n_classes) -> transpose to (n_classes, n_samples, n_features)
                shap_array = np.transpose(shap_array, (2, 0, 1))
                print(f"   Transposed to (n_classes, n_samples, n_features): {shap_array.shape}")
            elif shap_array.shape[1] > 1 and shap_array.shape[2] > 1:
                # Already (n_samples, n_classes, n_features) -> transpose to (n_classes, n_samples, n_features)
                shap_array = np.transpose(shap_array, (1, 0, 2))
                print(f"   Transposed to (n_classes, n_samples, n_features): {shap_array.shape}")
        n_classes, n_samples, n_features = shap_array.shape
    else:
        raise ValueError("Unrecognized SHAP output format")

    # Create feature names
    dims_per_band = [ts.transform(covs_reg[:1, b, :, :]).shape[1] for b, ts in enumerate(ts_list)]
    feature_names = []
    for b, dim in enumerate(dims_per_band):
        feature_names.extend([f"{band_names[b]}_f{i}" for i in range(dim)])
    print(f"   Generated {len(feature_names)} feature names.")

    # Compute mean absolute SHAP across all classes and samples
    # shap_array shape: (n_classes, n_samples, n_features)
    all_abs = np.abs(shap_array)                 # same shape
    mean_abs = all_abs.mean(axis=(0, 1))        # average over classes and samples -> (n_features,)
    print(f"   mean_abs shape: {mean_abs.shape}")

    # Verify lengths match
    if len(feature_names) != len(mean_abs):
        raise ValueError(f"Length mismatch: features={len(feature_names)}, mean_abs={len(mean_abs)}")

    # Save outputs
    print("9. Saving results...")
    # Save raw SHAP values (as transposed array)
    np.save(f"{OUT_PREFIX}_values_array.npy", shap_array)
    # Also save as list-of-arrays for compatibility
    shap_list = [shap_array[c, :, :] for c in range(n_classes)]
    joblib.dump(shap_list, f"{OUT_PREFIX}_values_list.pkl")

    # Feature importance CSV
    imp_df = pd.DataFrame({'feature': feature_names, 'mean_abs_shap': mean_abs})
    imp_df = imp_df.sort_values('mean_abs_shap', ascending=False)
    imp_df.to_csv(f"{OUT_PREFIX}_importance.csv", index=False)

    print(f"\n✅ Saved SHAP array to {OUT_PREFIX}_values_array.npy")
    print(f"✅ Saved SHAP list to {OUT_PREFIX}_values_list.pkl")
    print(f"✅ Saved feature importance to {OUT_PREFIX}_importance.csv")
    print("\nTop 10 features:")
    print(imp_df.head(10).to_string(index=False))

    # Optional: quick plot of top 20 features
    try:
        import matplotlib.pyplot as plt
        top20 = imp_df.head(20)
        plt.figure(figsize=(10, 6))
        plt.barh(top20['feature'], top20['mean_abs_shap'], color='steelblue')
        plt.xlabel("Mean |SHAP|")
        plt.title("Top 20 Tangent Space Features (XGBoost)")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f"{OUT_PREFIX}_top20.png", dpi=150)
        print(f"✅ Saved top‑20 plot to {OUT_PREFIX}_top20.png")
        plt.show()
    except ImportError:
        print("matplotlib not available – skipping plot.")

if __name__ == "__main__":
    main()