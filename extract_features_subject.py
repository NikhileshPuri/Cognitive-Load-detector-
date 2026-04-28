#!/usr/bin/env python3
# extract_features_subject_strict.py
"""
Strict extractor: subsets each preproc .npz to canonical channels in common_channels.txt.
If any .npz file for the subject does not contain ALL common channels, that file is skipped (no padding).
Usage:
  python extract_features_subject_strict.py --subject sub-28 --preproc_dir "D:/Data_cog/data_preproc" --out_dir "D:/Data_cog/features" --common "D:/Data_cog/features/common_channels.txt"
"""
import argparse, glob, os, time, numpy as np, pandas as pd
from scipy.signal import welch, butter, sosfiltfilt

bands = {'theta':(4,7), 'alpha':(8,12), 'beta':(13,30)}

def bandpass(epoch, sfreq, low, high, order=4):
    from scipy.signal import butter, sosfiltfilt
    sos = butter(order, [low, high], btype='bandpass', fs=sfreq, output='sos')
    return sosfiltfilt(sos, epoch, axis=-1)

def bandpower_linear(epoch, sfreq, band):
    from scipy.signal import welch
    f, Pxx = welch(epoch, fs=sfreq, nperseg=min(256, epoch.shape[-1]), axis=-1)
    if Pxx.shape[0] != f.shape[0] and Pxx.shape[1] == f.shape[0]:
        Pxx = Pxx.T
    sel = (f >= band[0]) & (f <= band[1])
    p = Pxx[sel, :].sum(axis=0)
    return p

def cov_epoch(epoch):
    return np.cov(epoch)

def load_common_channels(path):
    with open(path, 'r', encoding='utf8') as f:
        return [l.strip() for l in f if l.strip()]

def process_subject(subject, preproc_dir, out_dir, common_ch_file, timeout=300):
    common = load_common_channels(common_ch_file)
    pattern = os.path.join(preproc_dir, f"{subject}_*.npz")
    t0 = time.time()
    while True:
        files = sorted(glob.glob(pattern))
        if files:
            break
        if time.time() - t0 > timeout:
            raise TimeoutError("Timed out waiting for files for " + subject)
        time.sleep(1.0)
    print("Found", len(files), "files for", subject)
    rows=[]; covs=[]
    for fn in files:
        d = np.load(fn, allow_pickle=True)
        epochs = d.get('epochs', None)
        if epochs is None or epochs.size == 0:
            print(" Skipping (no epochs):", os.path.basename(fn)); continue
        chn = d.get('ch_names', None)
        if chn is None:
            print(" Skipping (no ch_names):", os.path.basename(fn)); continue
        chn = [c.decode() if isinstance(c, (bytes, np.bytes_)) else str(c) for c in chn]
        # Map common channels to file indices
        try:
            idxs = [chn.index(c) for c in common]
        except ValueError as e:
            print(" Skipping file (missing common channels):", os.path.basename(fn)); continue
        sfreq = float(d.get('sfreq', 128.0))
        task = str(d.get('task', os.path.splitext(os.path.basename(fn))[0]))
        # subset and process epochs
        for ei in range(epochs.shape[0]):
            ep = epochs[ei]
            if ep.ndim != 2:
                print("  skipping epoch (bad ndim):", ep.shape); continue
            # ensure ep has shape (n_ch, n_times)
            if ep.shape[0] != len(chn) and ep.shape[1] == len(chn):
                ep = ep.T
            # subset to common channel order
            ep = ep[np.array(idxs), :]
            n_ch, n_times = ep.shape
            bp_summary = {}
            covs_per_band = []
            linear_totals = []
            for bname, band in bands.items():
                epb = bandpass(ep, sfreq, band[0], band[1])
                covs_per_band.append(cov_epoch(epb).astype(np.float32))
                bp_lin = bandpower_linear(ep, sfreq, band)
                bp_log = np.log(bp_lin + 1e-12)
                bp_summary[f"{bname}_mean"] = float(bp_log.mean())
                bp_summary[f"{bname}_var"]  = float(bp_log.var())
                bp_summary[f"{bname}_mean_lin"] = float(bp_lin.mean())
                linear_totals.append(bp_lin.mean())
            lt = np.array(linear_totals, dtype=float)
            lt = np.clip(lt, 1e-12, None)
            props = lt / lt.sum()
            spec_ent = -float(np.sum(props * np.log(props)))
            row = {'subject':subject, 'session':str(d.get('session','ses-unknown')), 'task':task, 'epoch_idx':int(ei), 'label':task}
            row.update(bp_summary)
            row['spec_entropy_3band'] = spec_ent
            rows.append(row)
            covs.append(np.stack(covs_per_band, axis=0))
    if len(rows) == 0:
        print("No features extracted for", subject)
        return
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    out_csv = os.path.join(out_dir, f"{subject}_features.csv")
    out_cov = os.path.join(out_dir, f"covs_{subject}.npy")
    df.to_csv(out_csv, index=False)
    np.save(out_cov, np.stack(covs, axis=0))
    print("Saved features:", out_csv)
    print("Saved covs:", out_cov)
    return out_csv, out_cov

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--subject', required=True)
    p.add_argument('--preproc_dir', default=r"D:\Data_cog\data_preproc")
    p.add_argument('--out_dir', default=r"D:\Data_cog\features")
    p.add_argument('--common', default=r"D:\Data_cog\features\common_channels.txt")
    p.add_argument('--timeout', type=int, default=300)
    args = p.parse_args()
    process_subject(args.subject, args.preproc_dir, args.out_dir, args.common, timeout=args.timeout)
