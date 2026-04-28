#!/usr/bin/env python3
# map_and_smooth_preds.py
# Usage:
# python map_and_smooth_preds.py --in "D:/Data_cog/visualizations/sub-02_preds_v2.csv" --out_dir "D:/Data_cog/visualizations" --smooth 5

import argparse, os
import pandas as pd
import numpy as np

DEFAULT_MAP = {
    'zeroBACK':'Low', 'RS_Beg_EO':'Low', 'RS_Beg_EC':'Low', 'RS_End_EO':'Low', 'RS_End_Ec':'Low',
    'oneBACK':'Medium', 'Flanker':'Medium', 'MATBeasy':'Medium', 'PVT':'Medium',
    'twoBACK':'High', 'MATBmed':'High', 'MATBdiff':'High'
}

def load_df(path):
    df = pd.read_csv(path)
    return df

def compute_load_score(df, map_task=DEFAULT_MAP):
    # Ensure probabilites present
    for col in ('p_low','p_med','p_high'):
        if col not in df.columns:
            raise ValueError(f"Missing column {col} in predictions CSV")
    # map predicted task to discrete label (just for reference)
    df['pred_load_from_task'] = df['pred_task'].map(map_task).fillna('Unknown')
    # continuous load: 0.0(Low) - 0.5(Med) - 1.0(High)
    df['load_score'] = 0.5 * df['p_med'] + 1.0 * df['p_high']
    # discrete by thresholds
    df['load_label'] = np.where(df['load_score'] < 0.33, 'Low',
                          np.where(df['load_score'] < 0.66, 'Medium','High'))
    return df

def smooth_load_score(df, window=5):
    # simple centered moving average (pad with edge values)
    arr = df['load_score'].to_numpy(dtype=float)
    if window <= 1:
        df['load_score_smooth'] = arr
        df['load_label_smooth'] = df['load_label']
        return df
    kernel = np.ones(window) / window
    # pad by repeating ends to keep same length
    pad = window // 2
    arr_p = np.pad(arr, (pad, window - pad - 1), mode='edge')
    sm = np.convolve(arr_p, kernel, mode='valid')
    df['load_score_smooth'] = sm
    df['load_label_smooth'] = np.where(df['load_score_smooth'] < 0.33, 'Low',
                                np.where(df['load_score_smooth'] < 0.66, 'Medium','High'))
    return df

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--in', dest='infile', required=True)
    p.add_argument('--out_dir', default=r"D:\Data_cog\visualizations")
    p.add_argument('--smooth', type=int, default=5)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = load_df(args.infile)
    df = compute_load_score(df)
    out1 = os.path.join(args.out_dir, os.path.basename(args.infile).replace('.csv', '_mapped.csv'))
    df.to_csv(out1, index=False)
    print("Wrote:", out1)
    df = smooth_load_score(df, window=args.smooth)
    out2 = os.path.join(args.out_dir, os.path.basename(args.infile).replace('.csv', f'_mapped_sm{args.smooth}.csv'))
    df.to_csv(out2, index=False)
    print("Wrote:", out2)

if __name__ == '__main__':
    main()
