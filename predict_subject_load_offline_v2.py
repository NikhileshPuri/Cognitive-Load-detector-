#!/usr/bin/env python3
# D:\Data_cog\Scripts\predict_subject_load_offline_v2.py
import os, argparse, joblib, numpy as np, pandas as pd, time
from sklearn.preprocessing import StandardScaler

try:
    from pyriemann.tangentspace import TangentSpace
except Exception:
    TangentSpace = None

# default mapping from task -> load level (edit if you prefer)
DEFAULT_LOAD_MAP = {
    'twoBACK': 'high', 'MATBdiff': 'high',
    'oneBACK': 'medium', 'MATBmed': 'medium', 'Flanker': 'medium', 'PVT': 'medium',
    'zeroBACK': 'low', 'MATBeasy': 'low',
    'RS_Beg_EO': 'low', 'RS_Beg_EC': 'low', 'RS_End_EO': 'low', 'RS_End_Ec': 'low'
}

def find_ts_obj(pipeline):
    # Try known keys
    keys = ['ts_list','tslist','ts_list_','tangents','tangent_list','ts','tangentspace','ts_list']
    for k in keys:
        if k in pipeline and pipeline[k] is not None:
            return pipeline[k], k
    # fallback: find any key whose value is a list/tuple of pyriemann TangentSpace or has 'transform'
    for k,v in pipeline.items():
        if v is None:
            continue
        if isinstance(v, (list, tuple)) and len(v) > 0:
            # check element type quickly
            if hasattr(v[0], 'transform'):
                return v, k
        if hasattr(v, 'transform') and hasattr(v, 'fit'):
            # looks like a TangentSpace-like object
            return v, k
    return None, None

def regularize_covs(covs, reg):
    if reg is None or reg == 0:
        return covs
    covs_reg = covs.copy()
    n_epochs, n_bands, n_ch, _ = covs.shape
    I = np.eye(n_ch, dtype=covs.dtype)
    for i in range(n_epochs):
        for b in range(n_bands):
            covs_reg[i,b] = covs_reg[i,b] + reg * I
    return covs_reg

def transform_with_ts_list(covs, ts_obj):
    # covs: n_epochs x n_bands x n_ch x n_ch
    n_epochs, n_bands, n_ch, _ = covs.shape
    # if ts_obj is list-like, expect one TangentSpace per band
    if isinstance(ts_obj, (list, tuple)):
        Xbands = []
        for b, ts in enumerate(ts_obj):
            Xb = ts.transform(covs[:, b, :, :])
            Xbands.append(Xb)
        X = np.concatenate(Xbands, axis=1)
        return X
    # else assume ts_obj is fitted on stacked covs
    if hasattr(ts_obj, 'transform'):
        stacked = covs.reshape(n_epochs * n_bands, n_ch, n_ch)
        Xt = ts_obj.transform(stacked)
        X = Xt.reshape(n_epochs, n_bands * Xt.shape[1])
        return X
    raise RuntimeError("Unknown tangents object")

def infer_class_to_task(clf, all_features_csv):
    # try to infer mapping from task_enc to task string using all_features_csv
    mapping = {}
    if all_features_csv and os.path.exists(all_features_csv):
        df = pd.read_csv(all_features_csv)
        if 'task_enc' in df.columns and 'task' in df.columns:
            mapping = df.groupby('task_enc')['task'].first().to_dict()
    # build class index -> task name (clf.classes_ are encoded ints)
    idx2task = {}
    for idx, cls in enumerate(clf.classes_):
        idx2task[idx] = mapping.get(int(cls), str(int(cls)))
    return idx2task

def map_prob_to_load(prob_vec, idx2task, load_map):
    p_low = p_med = p_high = 0.0
    for i,p in enumerate(prob_vec):
        task = idx2task.get(i, None)
        label = load_map.get(task, 'low')
        if label == 'high': p_high += p
        elif label == 'medium': p_med += p
        else: p_low += p
    score = p_high*1.0 + p_med*0.5 + p_low*0.0
    return float(score), float(p_low), float(p_med), float(p_high)

def main(args):
    pipeline = joblib.load(args.pipeline)
    ts_obj, ts_key = find_ts_obj(pipeline)
    if ts_obj is not None:
        print("Found tangent object under key:", ts_key)
    else:
        print("No tangent object found in pipeline. Will fit TangentSpace(s) per-band on subject covs.")
    scaler = pipeline.get('scaler', None)
    clf = pipeline.get('classifier', pipeline.get('clf', None))
    if clf is None:
        raise RuntimeError("Classifier not found in pipeline.")

    covs = np.load(args.covs, allow_pickle=False)
    df = pd.read_csv(args.features_csv)
    print("Covs shape:", covs.shape, "CSV rows:", len(df))

    if ts_obj is None:
        # fit per-band TangentSpace(s) on subject covs
        if TangentSpace is None:
            raise RuntimeError("pyriemann is not installed; cannot fit tangents on subject.")
        print("Fitting TangentSpace(s) on subject covs (reg={})...".format(args.reg))
        covs = regularize_covs(covs, reg=args.reg)
        ts_list = []
        for b in range(covs.shape[1]):
            ts = TangentSpace(metric='riemann')
            ts.fit(covs[:, b, :, :])
            ts_list.append(ts)
        ts_obj = ts_list
        print("Fitted {} TangentSpace objects.".format(len(ts_list)))

    # regularize covs if requested before transform
    covs = regularize_covs(covs, reg=args.reg)

    t0 = time.time()
    X = transform_with_ts_list(covs, ts_obj)
    t1 = time.time()
    print("Tangent transform done in {:.3f}s shape {}".format(t1-t0, X.shape))

    if scaler is not None:
        X = scaler.transform(X)

    probs = clf.predict_proba(X)
    preds = clf.predict(X)

    idx2task = infer_class_to_task(clf, args.all_features_csv)
    load_map = DEFAULT_LOAD_MAP.copy()
    if args.load_map_csv and os.path.exists(args.load_map_csv):
        ldf = pd.read_csv(args.load_map_csv)
        for _,r in ldf.iterrows():
            load_map[str(r['task'])] = str(r['load'])

    rows=[]
    for i in range(len(probs)):
        score, pl, pm, ph = map_prob_to_load(probs[i], idx2task, load_map)
        # find predicted task name
        try:
            pred_int = int(preds[i])
            cls_idx = list(clf.classes_).index(pred_int)
            pred_task = idx2task.get(cls_idx, str(pred_int))
        except Exception:
            pred_task = str(preds[i])
        rows.append({
            'epoch_idx': int(i),
            'pred_label_int': int(preds[i]),
            'pred_task': pred_task,
            'score': score, 'p_low': pl, 'p_med': pm, 'p_high': ph
        })
    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print("Saved predictions to", args.out_csv)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("--pipeline", default=r"D:\Data_cog\models\tangent_agg.joblib")
    p.add_argument("--covs", required=True)
    p.add_argument("--features_csv", required=True)
    p.add_argument("--all_features_csv", default=r"D:\Data_cog\features\all_features_clean.csv")
    p.add_argument("--load_map_csv", default=None)
    p.add_argument("--out_csv", default=r"D:\Data_cog\visualizations\sub-predictions.csv")
    p.add_argument("--reg", type=float, default=1e-2)
    args = p.parse_args()
    main(args)
