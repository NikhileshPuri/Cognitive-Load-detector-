# D:\Data_cog\Scripts\preproc_all.py
"""
Preprocess .set/.fdt EEG files under D:\Data_cog and save per-file .npz to D:\Data_cog\data_preproc
Runs: notch 50Hz, bandpass 1-40Hz, resample 128Hz, ICA (picard/fastica), epoching:
 - N-back: 0-2s sliding non-overlap
 - MATB: 5s sliding non-overlap

Usage:
    python preproc_all.py --root "D:/Data_cog" --out "D:/Data_cog/data_preproc"
    python preproc_all.py --root "D:/Data_cog" --out "D:/Data_cog/data_preproc" --subject sub-28
"""
import os, argparse, glob, numpy as np, mne
from mne.preprocessing import ICA
from scipy.signal import butter, sosfiltfilt

def bandpass_sos(data, sfreq, low, high, order=4):
    sos = butter(order, [low, high], btype='bandpass', fs=sfreq, output='sos')
    return sosfiltfilt(sos, data, axis=-1)

def epoch_sliding(data, sfreq, epoch_sec, step_sec):
    n_ch, n_times = data.shape
    step = int(step_sec * sfreq)
    win = int(epoch_sec * sfreq)
    epochs=[]
    for s in range(0, n_times - win + 1, step):
        epochs.append(data[:, s:s+win])
    if len(epochs)==0:
        return np.zeros((0, n_ch, win), dtype=np.float32)
    return np.stack(epochs, axis=0)

def preprocess_file(setfile, outdir):
    try:
        raw = mne.io.read_raw_eeglab(setfile, preload=True, verbose='ERROR')
    except Exception as e:
        print("Failed to read", setfile, e); return
    raw.pick_types(eeg=True)
    # drop ECG channel if present (named 'ECG' in dataset)
    if 'ECG' in raw.ch_names:
        try:
            raw.drop_channels(['ECG'])
        except Exception:
            pass
    try:
        raw.notch_filter(freqs=50.0, verbose='ERROR')
    except Exception:
        pass
    try:
        raw.filter(1., 40., method='fir', verbose='ERROR')
    except Exception:
        pass
    try:
        raw.resample(128, verbose='ERROR')
    except Exception:
        pass

    sfreq = raw.info['sfreq']
    ch_names = raw.ch_names

    # ICA
    try:
        n_comp = max(1, raw.info['nchan'] - 1)
        ica = ICA(n_components=min(n_comp, 32), method='picard' if 'picard' in dir(ICA) else 'fastica', random_state=42)
        ica.fit(raw)
        # try EOG auto detect only if EOG channel exists
        if any('EOG' in ch for ch in raw.ch_names):
            eog_inds, _ = ica.find_bads_eog(raw)
            ica.exclude = eog_inds
        raw = ica.apply(raw.copy())
    except Exception as e:
        print("ICA skipped/failed for", setfile, e)

    data = raw.get_data()  # n_ch x n_times

    # determine task name from filename and choose epoching
    fname = os.path.basename(setfile)
    name = os.path.splitext(fname)[0]
    if 'MATB' in name.upper():
        epoch_sec = 5.0; step_sec = 5.0
    else:
        epoch_sec = 2.0; step_sec = 2.0

    epochs = epoch_sliding(data, sfreq, epoch_sec, step_sec)  # n_epochs x n_ch x n_times
    subj = "unknown"
    parts = setfile.replace("\\","/").split("/")
    for p in parts:
        if p.startswith('sub-') or p.startswith('sub_') or p.startswith('sub'):
            subj = p
    session = 'ses-unknown'
    for p in parts:
        if p.startswith('ses-') or p.startswith('ses_') or p.startswith('ses'):
            session = p
    outfn = os.path.join(outdir, f"{subj}_{session}_{name}.npz")
    np.savez_compressed(outfn, epochs=epochs.astype(np.float32), ch_names=np.array(ch_names), sfreq=np.float32(sfreq), subject=subj, session=session, task=name)
    print("Saved", outfn)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default=r"D:\Data_cog")
    p.add_argument('--out', default=r"D:\Data_cog\data_preproc")
    p.add_argument('--subject', default=None, help="If provided (e.g. sub-28), only preprocess files under that subject")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    set_files = glob.glob(os.path.join(args.root, "sub-*", "ses-*", "eeg", "*.set"))
    set_files += glob.glob(os.path.join(args.root, "sub-*", "ses-*", "eeg", "*.SET"))
    if args.subject:
        # filter file list to only those that contain the subject string in their path
        set_files = [f for f in set_files if ("/" + args.subject + "/").replace("/", os.sep) in f or ("\\" + args.subject + "\\").replace("\\", os.sep) in f or os.path.basename(f).startswith(args.subject + "_")]
    print("Found", len(set_files), ".set files (after subject filter)")
    for f in sorted(set_files):
        try:
            preprocess_file(f, args.out)
        except Exception as e:
            print("Error preprocessing", f, e)

if __name__ == "__main__":
    main()
