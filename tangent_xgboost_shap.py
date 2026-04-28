#!/usr/bin/env python3
"""
tangent_xgboost_shap.py

Usage example:
python tangent_xgboost_shap.py \
  --all_csv "D:/Data_cog/features/all_features_clean.csv" \
  --all_covs "D:/Data_cog/features/all_covs.npy" \
  --train_csv "D:/Data_cog/features/splits/train.csv" \
  --test_csv  "D:/Data_cog/features/splits/test.csv" \
  --val_csv   "D:/Data_cog/features/splits/val.csv" \
  --out_model "D:/Data_cog/models/tangent_xgb.joblib" \
  --reg 0.01 --nrounds 200 --early_stopping 20 --compute_shap --shap_sample 500
"""
import os
import time
import argparse
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

# pyriemann
from pyriemann.tangentspace import TangentSpace

# sklearn utilities
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix

# xgboost
import xgboost as xgb

# optional shap
try:
    import shap
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False

def regularize_covs(covs, reg):
    # covs: (n_epochs, n_bands, n_ch, n_ch)
    n_epochs, n_bands, n_ch, _ = covs.shape
    if reg == 0:
        return covs
    I = np.eye(n_ch, dtype=covs.dtype) * reg
    covs_reg = covs.copy()
    for b in range(n_bands):
        covs_reg[:, b, :, :] = covs_reg[:, b, :, :] + I
    return covs_reg

def fit_ts_per_band(covs, idx_train, band_names, reg, verbose=True):
    """
    Fit a TangentSpace object per band on training covariances.
    covs: mmap or numpy array shape (n_epochs, n_bands, n_ch, n_ch)
    idx_train: indices of training epochs into covs
    returns: list of fitted TangentSpace objects, transforms dims list
    """
    n_bands = covs.shape[1]
    ts_list = []
    dims = []
    if verbose:
        print("Fitting TangentSpace per band on train covs...")
    for b in range(n_bands):
        t0 = time.time()
        stacked = covs[idx_train, b, :, :]  # shape (n_train, n_ch, n_ch)
        ts = TangentSpace(metric='riemann')
        ts.fit(stacked)
        Xt = ts.transform(stacked)
        dims.append(Xt.shape[1])
        ts_list.append(ts)
        if verbose:
            print(f"  fitted TS band {b} ('{band_names[b] if b < len(band_names) else b}') -> dim {dims[-1]}  time {time.time()-t0:.2f}s")
    return ts_list, dims

def transform_all_bands(covs, ts_list, band_names, verbose=True):
    """
    Transform all covs per band using ts_list; returns X_all shape (n_epochs, total_dim)
    """
    n_epochs, n_bands, n_ch, _ = covs.shape
    X_bands = []
    if verbose:
        print("Transforming covariances to tangent features (per band)...")
    for b, ts in enumerate(ts_list):
        t0 = time.time()
        # transform whole column for band b
        with np.errstate(all='ignore'):
            Xb = ts.transform(covs[:, b, :, :])  # shape (n_epochs, d_b)
        X_bands.append(Xb)
        if verbose:
            print(f"  band {b} ('{band_names[b] if b < len(band_names) else b}') transformed -> {Xb.shape}  time {time.time()-t0:.2f}s")
    X_all = np.concatenate(X_bands, axis=1)
    if verbose:
        print("Concatenated feature shape:", X_all.shape)
    return X_all

def train_xgb_sklearn(Xtr, ytr, Xeval, yeval, args):
    """
    Train using sklearn XGBClassifier wrapper with early stopping (requires eval_set)
    Returns classifier object (sklearn wrapper).
    """
    from xgboost import XGBClassifier
    n_classes = int(np.unique(ytr).size)
    clf = XGBClassifier(
        objective='multi:softprob',
        num_class=n_classes,
        n_estimators=args.nrounds,
        use_label_encoder=False,
        n_jobs=-1,
        verbosity=1,
        eval_metric='mlogloss'
    )
    print("Training XGBoost sklearn-wrapper with early stopping (eval_set provided)...")
    start = time.time()
    clf.fit(
        Xtr, ytr,
        eval_set=[(Xeval, yeval)],
        early_stopping_rounds=args.early_stopping,
        verbose=True
    )
    print("Sklearn XGB training finished in {:.1f}s, best_ntree_limit: {}".format(time.time()-start,
                                                                                   getattr(clf, "best_ntree_limit", "N/A")))
    return clf

def train_xgb_core(Xtr, ytr, Xeval, yeval, args):
    """
    Train using xgboost.train lower-level API (DMatrix + watchlist).
    Returns booster object.
    """
    print("Training XGBoost with xgb.train fallback (DMatrix + watchlist)...")
    dtrain = xgb.DMatrix(Xtr, label=ytr)
    dvalid = xgb.DMatrix(Xeval, label=yeval)
    params = {
        'objective': 'multi:softprob',
        'num_class': int(np.unique(ytr).size),
        'eval_metric': 'mlogloss',
        'n_jobs': -1,
        'verbosity': 1
    }
    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
    start = time.time()
    bst = xgb.train(params, dtrain, num_boost_round=args.nrounds,
                    evals=watchlist, early_stopping_rounds=args.early_stopping,
                    verbose_eval=True)
    print("xgb.train finished in {:.1f}s, best_iteration: {}".format(time.time()-start, bst.best_iteration))
    return bst

def compute_shap_and_save(clf_obj, X_sample, feature_names, out_prefix):
    """
    Compute SHAP values (TreeExplainer) for clf_obj.
    clf_obj can be sklearn XGBClassifier or xgboost.Booster.
    Saves shap_values (n_samples x n_features) -> npy and per-feature mean_abs shap importance csv.
    """
    if not _HAS_SHAP:
        print("SHAP not installed; skipping SHAP computation. Install shap via `pip install shap` to enable.")
        return None
    print("Computing SHAP values on sample size:", X_sample.shape[0])
    explainer = shap.TreeExplainer(clf_obj)
    shap_vals = explainer.shap_values(X_sample)  # for multiclass returns list of arrays or array
    # Convert to per-feature importance (mean abs across classes & samples)
    if isinstance(shap_vals, list):
        # shap_vals: list[class] -> (n_samples, n_features). We'll average absolute across classes & samples
        vals = np.stack([np.abs(v) for v in shap_vals], axis=0)  # (n_classes, n_samples, n_feat)
        mean_abs = vals.mean(axis=(0,1))
    else:
        mean_abs = np.abs(shap_vals).mean(axis=0)
    out_vals = out_prefix + "_shap_values.npy"
    out_imp = out_prefix + "_shap_feature_importance.csv"
    np.save(out_vals, shap_vals)
    pd.DataFrame({'feature': feature_names, 'mean_abs_shap': mean_abs}).sort_values('mean_abs_shap', ascending=False).to_csv(out_imp, index=False)
    print("Saved SHAP values to:", out_vals)
    print("Saved SHAP feature importance to:", out_imp)
    return out_vals, out_imp

def main(args):
    # Load CSV & covs
    print("Loading CSV and index mapping...")
    df_all = pd.read_csv(args.all_csv)
    covs = np.load(args.all_covs, mmap_mode='r')  # (n_epochs, n_bands, n_ch, n_ch)
    print("CSV rows:", len(df_all), " Covs shape:", covs.shape, " dtype:", covs.dtype)

    # check shape
    if covs.shape[0] != len(df_all):
        raise ValueError(f"Mismatch: covs first-dim {covs.shape[0]} != csv rows {len(df_all)}")

    # load splits
    train_df = pd.read_csv(args.train_csv)
    test_df  = pd.read_csv(args.test_csv)
    val_df   = pd.read_csv(args.val_csv) if args.val_csv and os.path.exists(args.val_csv) else None

    print(f"Train samples: {len(train_df)} Test samples: {len(test_df)} Val samples: {len(val_df) if val_df is not None else 0}")

    uid_to_idx = {uid: i for i, uid in enumerate(df_all['uid'].astype(str).values)}
    def get_idxs(df):
        return np.array([uid_to_idx[u] for u in df['uid'].astype(str).values if u in uid_to_idx], dtype=int)

    idx_train = get_idxs(train_df)
    idx_test  = get_idxs(test_df)
    idx_val   = get_idxs(val_df) if val_df is not None else None

    if idx_train.size == 0 or idx_test.size == 0:
        raise ValueError("Empty train or test indices; check uid matching")

    # regularize covs if requested
    if args.reg > 0:
        print("Regularizing covariances with reg =", args.reg)
        # we will apply regularization on-the-fly when fitting/transforming by adding reg * I in place.
        covs_reg = regularize_covs(covs, args.reg)
    else:
        covs_reg = covs

    # prepare band names
    band_names = ['theta','alpha','beta'] if covs.shape[1] == 3 else [f'band{b}' for b in range(covs.shape[1])]

    # Fit TangentSpace per band on train only
    ts_list, dims = fit_ts_per_band(covs_reg, idx_train, band_names, args.reg, verbose=True)
    total_dim = sum(dims)

    # Transform all covs (per band) -> X_all
    X_all = transform_all_bands(covs_reg, ts_list, band_names, verbose=True)

    # labels
    if 'task_enc' not in df_all.columns:
        raise ValueError("all_csv must contain 'task_enc' encoded labels.")
    y_all = df_all['task_enc'].values.astype(int)

    # Prepare splits
    Xtr, ytr = X_all[idx_train], y_all[idx_train]
    Xte, yte = X_all[idx_test],  y_all[idx_test]
    if idx_val is not None:
        Xval, yval = X_all[idx_val], y_all[idx_val]
    else:
        Xval, yval = None, None

    # scaling if requested
    if args.scale:
        scaler = StandardScaler()
        print("Scaling features (StandardScaler)...")
        Xtr = scaler.fit_transform(Xtr)
        Xte = scaler.transform(Xte)
        if Xval is not None:
            Xval = scaler.transform(Xval)
    else:
        scaler = None

    # training: try sklearn wrapper first, fallback to xgb.train
    clf = None
    fallback_used = False
    try:
        # sklearn wrapper expects eval_set. choose validation if available else test.
        Xeval, yeval = (Xval, yval) if (Xval is not None and Xval.shape[0] > 0) else (Xte, yte)
        clf = train_xgb_sklearn(Xtr, ytr, Xeval, yeval, args)
    except Exception as e:
        print("=== sklearn XGBClassifier training failed with error ===")
        print(e)
        print("Activating fallback: training with xgboost.train (DMatrix + watchlist).")
        fallback_used = True
        # use validation if available else test
        Xeval, yeval = (Xval, yval) if (Xval is not None and Xval.shape[0] > 0) else (Xte, yte)
        bst = train_xgb_core(Xtr, ytr, Xeval, yeval, args)
        clf = bst  # store booster in clf variable

    # Evaluate
    print("Evaluating on test set...")
    if not fallback_used:
        # sklearn wrapper
        ypred = clf.predict(Xte)
        yprob = clf.predict_proba(Xte)
    else:
        # booster: predict proba via DMatrix
        dtest = xgb.DMatrix(Xte)
        yprob = clf.predict(dtest)  # (n_samples, n_classes)
        ypred = np.argmax(yprob, axis=1)

    print("Test accuracy:", accuracy_score(yte, ypred))
    print("Test F1 (macro):", f1_score(yte, ypred, average='macro'))
    print("Classification report:\n", classification_report(yte, ypred))
    print("Confusion matrix:\n", confusion_matrix(yte, ypred))

    # Optionally compute SHAP (on sample subset)
    shap_info = None
    if args.compute_shap:
        if not _HAS_SHAP:
            print("SHAP not installed. Install shap with `pip install shap` to compute SHAP values.")
        else:
            # sample subset to limit cost
            nsample = min(args.shap_sample, Xtr.shape[0])
            rng = np.random.default_rng(42)
            sample_idx = rng.choice(Xtr.shape[0], size=nsample, replace=False)
            X_sample = Xtr[sample_idx]
            # feature names
            feat_names = [f"b{b}_f{i}" for b, d in enumerate(dims) for i in range(d)]
            try:
                if not fallback_used:
                    shap_prefix = os.path.splitext(args.out_model)[0] + "_xgb"
                    compute_shap_and_save(clf, X_sample, feat_names, shap_prefix)
                else:
                    # for booster, TreeExplainer accepts booster
                    shap_prefix = os.path.splitext(args.out_model)[0] + "_xgb_booster"
                    compute_shap_and_save(clf, X_sample, feat_names, shap_prefix)
            except Exception as e:
                print("SHAP computation failed:", e)

    # Save pipeline
    pipeline = {
        'ts_list': ts_list,
        'scaler': scaler,
        'classifier': clf,
        'reg': args.reg,
        'band_names': band_names,
        'feature_shape': total_dim,
        'df_all_len': len(df_all),
        'fallback_used': fallback_used
    }
    os.makedirs(os.path.dirname(args.out_model), exist_ok=True)
    joblib.dump(pipeline, args.out_model)
    print("Saved model pipeline to", args.out_model)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--all_csv", required=True)
    p.add_argument("--all_covs", required=True)
    p.add_argument("--train_csv", required=True)
    p.add_argument("--test_csv", required=True)
    p.add_argument("--val_csv", default="")
    p.add_argument("--out_model", default=r"D:\Data_cog\models\tangent_xgb.joblib")
    p.add_argument("--reg", type=float, default=1e-2)
    p.add_argument("--nrounds", type=int, default=200)
    p.add_argument("--early_stopping", type=int, default=20)
    p.add_argument("--scale", action='store_true')
    p.add_argument("--compute_shap", action='store_true')
    p.add_argument("--shap_sample", type=int, default=500, help="SHAP sample size")
    args = p.parse_args()
    main(args)
