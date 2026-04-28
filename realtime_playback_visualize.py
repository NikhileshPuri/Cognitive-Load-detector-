#!/usr/bin/env python3
# realtime_playback_visualize.py
# Usage:
# python realtime_playback_visualize.py --preds "D:/Data_cog/visualizations/sub-02_preds_v2_mapped_sm7.csv" --rate 1.0

import argparse, time, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def load_preds(path):
    df = pd.read_csv(path)
    if 'load_score_smooth' not in df.columns:
        raise ValueError("Run map_and_smooth_preds.py first (need load_score_smooth)")
    return df

def color_for_label(lbl):
    return {'Low':'#2ca02c','Medium':'#ff7f0e','High':'#d62728'}.get(lbl, '#7f7f7f')

def run_live(df, rate=1.0):
    # rate = seconds per epoch playback
    n = len(df)
    x = np.arange(n)
    y = df['load_score_smooth'].values
    labels = df['load_label_smooth'].values

    fig, ax = plt.subplots(figsize=(10,4))
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, max(50, n))
    ax.set_xlabel('Epoch index')
    ax.set_ylabel('Load score (0 low → 1 high)')
    line, = ax.plot([], [], lw=2)
    scatter = ax.scatter([], [], s=50)
    current_text = ax.text(0.02, 0.9, '', transform=ax.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

    # background colored bands for Low/Med/High
    ax.axhspan(0.0, 0.33, color='#dff0d8', alpha=0.3)
    ax.axhspan(0.33, 0.66, color='#fff4cc', alpha=0.3)
    ax.axhspan(0.66, 1.0, color='#f8d7da', alpha=0.3)

    def init():
        line.set_data([], [])
        scatter.set_offsets(np.empty((0,2)))
        current_text.set_text('')
        return line, scatter, current_text

    i_state = {'i':0}
    def update(frame):
        i = frame
        window = min(n, max(200, i+1))  # auto expand x limit if necessary
        ax.set_xlim(max(0, i-200), max(50, i+50))
        line.set_data(x[:i+1], y[:i+1])
        scatter.set_offsets(np.column_stack([ [i], [y[i]] ]))
        scatter.set_color(color_for_label(labels[i]))
        current_text.set_text(f"epoch {i}  load_score={y[i]:.3f}  label={labels[i]}")
        return line, scatter, current_text

    ani = FuncAnimation(fig, update, frames=range(n), init_func=init, interval=rate*1000, blit=False, repeat=False)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--preds', required=True)
    p.add_argument('--rate', type=float, default=1.0, help='seconds per epoch (playback rate)')
    args = p.parse_args()
    df = load_preds(args.preds)
    run_live(df, rate=args.rate)
