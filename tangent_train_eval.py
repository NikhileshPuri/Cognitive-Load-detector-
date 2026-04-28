#!/usr/bin/env python3
# D:\Data_cog\Scripts\tangent_train_eval.py
import os, argparse, time, joblib, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from pyriemann.tangentspace import TangentSpace

def regularize_slice(cov_slice, reg):
    # cov_slice: (n_samples, n_ch, n_ch)
    if reg is None or reg == 0:
        return cov_slice
    # add reg*I efficiently
    n, C, _ = cov_slice.shape
    cov_slice = cov_slice.copy()
    I = np.eye(C, dtype=cov_slice.dtype)
    for i in range(n):
        cov_slice[i] += reg * I
    return cov_slice

def slice_covs_memmap(all_covs, idxs, band):
    # all_covs is a memmap or ndarray with shape (N, B, C, C)
    # returns (len(idxs), C, C) view/copy
    return all_covs[idxs, band, :, :]

def fit_transform_per_band(all_covs, idx_train, idx_test, idx_val, reg, band_names, verbose=1):
    """
    For each band:
      - fit TangentSpace on training covs for that band (regularized)
      - transform train/test/val covs for that band
    Returns: Xtr (n_tr x F), Xte, Xval (or None), ts_list
    """
    N, B, C, _ = all_covs.shape
    ts_list = []
    Xtr_bands, Xte_bands, Xval_bands = [], [], []
    for b in range(B):
        bn = band_names[b] if b < len(band_names) else f"band{b}"
        if verbose:
            print(f"[band {b}] {bn} - shapes: (C={C}) - fitting TangentSpace on {len(idx_train)} train covs")
        covs_train = slice_covs_memmap(all_covs, idx_train, b)
        covs_train = regularize_slice(covs_train, reg)
        ts = TangentSpace(metric='riemann')
        t0 = time.time()
        ts.fit(covs_train)
        tfit = time.time() - t0
        if verbose:
            print(f"  fitted TS on band {b} in {tfit:.2f}s")
        # transform train/test/val
        Xtr_b = ts.transform(covs_train)            # (n_tr, feat_b)
        covs_test = slice_covs_memmap(all_covs, idx_test, b)
        covs_test = regularize_slice(covs_test, reg)
        Xte_b = ts.transform(covs_test)
        if idx_val is not None:
            covs_val = slice_covs_memmap(all_covs, idx_val, b)
            covs_val = regularize_slice(covs_val, reg)
            Xval_b = ts.transform(covs_val)
        else:
            Xval_b = None
        ts_list.append(ts)
        Xtr_bands.append(Xtr_b)
        Xte_bands.append(Xte_b)
        Xval_bands.append(Xval_b)
        if verbose:
            print(f"  band {b} shapes -> train {Xtr_b.shape}, test {Xte_b.shape}" + (f", val {Xval_b.shape}" if Xval_b is not None else ""))
    # concatenate across bands
    Xtr = np.concatenate(Xtr_bands, axis=1)
    Xte = np.concatenate(Xte_bands, axis=1)
    Xval = np.concatenate(Xval_bands, axis=1) if idx_val is not None else None
    return Xtr, Xte, Xval, ts_list

def load_csv_and_get_idxs(all_csv, train_csv, test_csv, val_csv):
    df_all = pd.read_csv(all_csv)
    train_df = pd.read_csv(train_csv)
    test_df  = pd.read_csv(test_csv)
    val_df = pd.read_csv(val_csv) if (val_csv and os.path.exists(val_csv)) else None
    uid_to_idx = {u: i for i,u in enumerate(df_all['uid'].astype(str).values)}
    def idxs_from_df(df):
        uids = df['uid'].astype(str).values
        idxs = [uid_to_idx[u] for u in uids if u in uid_to_idx]
        return np.array(idxs, dtype=int)
    idx_train = idxs_from_df(train_df)
    idx_test  = idxs_from_df(test_df)
    idx_val   = idxs_from_df(val_df) if val_df is not None else None
    return df_all, idx_train, idx_test, idx_val

def main(all_csv, all_covs_path, train_csv, test_csv, val_csv, out_model, reg=1e-2, scale=False, classifier='lr', band_names=None, verbose=1):
    t0_all = time.time()
    print("Loading CSV and index mapping...")
    df_all, idx_train, idx_test, idx_val = load_csv_and_get_idxs(all_csv, train_csv, test_csv, val_csv)
    print("Train samples:", len(idx_train), "Test samples:", len(idx_test), "Val samples:", 0 if idx_val is None else len(idx_val))

    if len(idx_train)==0 or len(idx_test)==0:
        raise ValueError("Empty train or test indices — check uid matching between CSV and splits")

    # load covs as memmap to avoid copying large arrays into RAM
    print("Loading covariances (mmap)...")
    all_covs = np.load(all_covs_path, mmap_mode='r')
    if all_covs.ndim != 4:
        raise ValueError("Expected covs array shape (N, B, C, C). Got: " + str(all_covs.shape))
    N, B, C, _ = all_covs.shape
    print("Covs shape:", all_covs.shape)

    # labels
    if 'task_enc' not in df_all.columns:
        raise ValueError("Expected 'task_enc' in CSV (encoded labels).")
    y_all = df_all['task_enc'].astype(int).values
    ytr = y_all[idx_train]
    yte = y_all[idx_test]
    yval = y_all[idx_val] if idx_val is not None else None

    # check class presence in train
    unique_train = np.unique(ytr)
    if unique_train.size < 2:
        raise ValueError(f"Training set contains only one class: {unique_train}. Cannot train classifier.")

    # default band names
    if band_names is None:
        band_names = [f"band{b}" for b in range(B)]

    # compute tangents per band (fit on train only)
    print("Computing tangent-space embeddings per band (fit on train only)...")
    t0 = time.time()
    Xtr, Xte, Xval, ts_list = fit_transform_per_band(all_covs, idx_train, idx_test, idx_val, reg, band_names, verbose=verbose)
    print(f"Tangent transforms done in {time.time()-t0:.2f}s -> shapes train {Xtr.shape}, test {Xte.shape}")

    # optional scaling
    scaler = None
    if scale:
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr)
        Xte = scaler.transform(Xte)
        if Xval is not None:
            Xval = scaler.transform(Xval)

    # classifier selection
    if classifier == 'lr':
        clf = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs', n_jobs=1)
    else:
        clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)

    print("Training classifier...")
    t0 = time.time()
    clf.fit(Xtr, ytr)
    tr_time = time.time()-t0
    print(f"Classifier trained in {tr_time:.2f}s")

    # evaluate
    pred = clf.predict(Xte)
    print("Test accuracy:", accuracy_score(yte, pred))
    print("Test F1 (macro):", f1_score(yte, pred, average='macro'))
    print("Confusion matrix:\n", confusion_matrix(yte, pred))
    print("Classification report:\n", classification_report(yte, pred))

    # optionally evaluate val
    if Xval is not None:
        pval = clf.predict(Xval)
        print("Val accuracy:", accuracy_score(yval, pval))
        print("Val F1:", f1_score(yval, pval, average='macro'))

    # Save pipeline: include ts_list (one per band), scaler, classifier, meta
    out = {
        'ts_list': ts_list,
        'scaler': scaler,
        'classifier': clf,
        'reg': reg,
        'band_names': band_names,
        'feature_shape': Xtr.shape[1],
        'df_all_len': len(df_all)
    }
    joblib.dump(out, out_model)
    print("Saved model pipeline to", out_model)
    print("Total runtime:", time.time() - t0_all)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--all_csv", default=r"D:\Data_cog\features\all_features_clean.csv")
    p.add_argument("--all_covs", default=r"D:\Data_cog\features\all_covs.npy")
    p.add_argument("--train_csv", default=r"D:\Data_cog\features\splits\train.csv")
    p.add_argument("--test_csv", default=r"D:\Data_cog\features\splits\test.csv")
    p.add_argument("--val_csv", default=r"D:\Data_cog\features\splits\val.csv")
    p.add_argument("--out", default=r"D:\Data_cog\models\tangent_agg.joblib")
    p.add_argument("--reg", type=float, default=1e-2)
    p.add_argument("--scale", action='store_true')
    p.add_argument("--classifier", default='lr', choices=['lr','rf'])
    p.add_argument("--verbose", type=int, default=1)
    args = p.parse_args()
    main(args.all_csv, args.all_covs, args.train_csv, args.test_csv, args.val_csv, args.out, reg=args.reg, scale=args.scale, classifier=args.classifier, verbose=args.verbose)
