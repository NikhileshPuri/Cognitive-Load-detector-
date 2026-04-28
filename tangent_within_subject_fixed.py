#!/usr/bin/env python3
"""
Tangent-space within-subject CV with adaptive regularization.

Usage:
python tangent_within_subject_fixed.py --covs D:/Data_cog/features/covs_sub-02.npy --features_csv D:/Data_cog/features/sub-02_features.csv --reg 1e-2
"""
import numpy as np, pandas as pd, argparse, os
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
import numpy.linalg as LA

p = argparse.ArgumentParser()
p.add_argument('--covs', default=r"D:\Data_cog\features\covs_sub-02.npy")
p.add_argument('--features_csv', default=r"D:\Data_cog\features\sub-02_features.csv")
p.add_argument('--n_splits', type=int, default=5)
p.add_argument('--reg', type=float, default=1e-2, help="regularization factor (eps = reg * mean(diag(C)))")
args = p.parse_args()

covs = np.load(args.covs)  # n_epochs x n_bands x n_ch x n_ch
df = pd.read_csv(args.features_csv)

# Simple label mapping to binary easy/difficult as before
def map_label(task):
    if 'MATBeasy' in task or 'MATBmed' in task: return 'easy'
    if 'MATBdiff' in task: return 'difficult'
    if 'zeroBACK' in task: return 'easy'
    if 'oneBACK' in task: return 'easy'
    if 'twoBACK' in task: return 'difficult'
    return task
labels = df['task'].map(map_label).values

n_epochs, n_bands, n_ch, _ = covs.shape
print("Loaded covs:", covs.shape, "labels:", labels.shape)
good_idx = []
covs_reg = []

for i in range(n_epochs):
    # stack per-band reg covs into shape n_bands x n_ch x n_ch
    entry = covs[i]
    ok = True
    entry_reg = []
    for b in range(n_bands):
        C = entry[b]
        if not np.isfinite(C).all():
            ok = False; break
        # symmetrize
        C = 0.5*(C + C.T)
        # compute adaptive eps
        eps = args.reg * (np.trace(C) / float(n_ch) + 1e-12)
        # add eps to diagonal
        C_reg = C + eps * np.eye(n_ch)
        # final check: if still not PD, add larger eps
        try:
            # smallest eigenvalue
            min_eig = np.min(LA.eigvalsh(C_reg))
            if min_eig <= 0:
                # increase epsilon multiplicatively until PD (few iterations)
                factor = 1.0
                while min_eig <= 0 and factor < 1e6:
                    factor *= 10.0
                    C_reg = C + (eps * factor) * np.eye(n_ch)
                    min_eig = np.min(LA.eigvalsh(C_reg))
                if min_eig <= 0:
                    ok = False
                    break
        except Exception:
            ok = False
            break
        entry_reg.append(C_reg)
    if ok:
        good_idx.append(i)
        covs_reg.append(np.stack(entry_reg, axis=0))
    else:
        # drop this epoch
        continue

if len(good_idx) == 0:
    raise RuntimeError("No valid covariance entries after regularization. Increase --reg or inspect covs.")

covs_reg = np.stack(covs_reg, axis=0)  # n_good x n_bands x n_ch x n_ch
labels_good = labels[good_idx]
print("Kept epochs:", covs_reg.shape[0], "out of", n_epochs)

# Build tangent features (concatenate per-band)
parts = []
for b in range(covs_reg.shape[1]):
    ts = TangentSpace().fit(covs_reg[:, b])
    parts.append(ts.transform(covs_reg[:, b]))
X = np.concatenate(parts, axis=1)

# 5-fold CV
skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
accs=[]; f1s=[]
for tr,te in skf.split(X, labels_good):
    clf = LogisticRegression(max_iter=2000, class_weight='balanced', solver='lbfgs')
    clf.fit(X[tr], labels_good[tr])
    pred = clf.predict(X[te])
    accs.append(accuracy_score(labels_good[te], pred))
    f1s.append(f1_score(labels_good[te], pred, average='macro'))
print("Tangent (reg={}) mean acc: {:.3f} mean f1: {:.3f}".format(args.reg, np.mean(accs), np.mean(f1s)))
