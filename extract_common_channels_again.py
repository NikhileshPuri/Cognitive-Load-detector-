#!/usr/bin/env python3
# write_common_channels.py
import glob, os, numpy as np
preproc = r"D:\Data_cog\data_preproc"
out = r"D:\Data_cog\features\common_channels.txt"

files = sorted(glob.glob(os.path.join(preproc, "*.npz")))
if not files:
    raise SystemExit("No preproc npz files found in " + preproc)

sets = []
for f in files:
    try:
        d = np.load(f, allow_pickle=True)
    except Exception as e:
        print("WARN: failed load", f, e); continue
    ch = d.get('ch_names', None)
    if ch is None:
        continue
    ch = [c.decode() if isinstance(c, (bytes, np.bytes_)) else str(c) for c in ch]
    sets.append(set(ch))

if not sets:
    raise SystemExit("No channel info found in any preproc files")

inter = sorted(list(set.intersection(*sets)))
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w", encoding="utf8") as fo:
    for c in inter:
        fo.write(c + "\n")
print("Wrote", out, "with", len(inter), "channels")
