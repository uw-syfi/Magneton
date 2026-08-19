#!/usr/bin/env python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', required=True, help='path to input CSV file')
parser.add_argument('-o', '--output', help='path to output image file')

# Assumes columns: timestamp, power(buggy), timestamp, power(normal)
def plot_power_compare(args):
    df = pd.read_csv(args.input)
 
    if df.shape[1] >= 4:
        df.columns = ['timestamp_buggy', 'power_buggy', 'timestamp_normal', 'power_normal']
    # Convert to numeric
    df['timestamp_buggy'] = pd.to_numeric(df['timestamp_buggy'], errors='coerce') / 1e6  # us to s
    df['timestamp_normal'] = pd.to_numeric(df['timestamp_normal'], errors='coerce') / 1e6  # us to s
    df['power_buggy'] = pd.to_numeric(df['power_buggy'], errors='coerce') / 1e3  # W to MW
    df['power_normal'] = pd.to_numeric(df['power_normal'], errors='coerce') / 1e3  # W to MW
    all_timestamps = np.union1d(df['timestamp_buggy'].dropna().values, df['timestamp_normal'].dropna().values)
    buggy_interp = np.interp(all_timestamps, df['timestamp_buggy'], df['power_buggy'])
    normal_interp = np.interp(all_timestamps, df['timestamp_normal'], df['power_normal'])
    min_power = 72.154
    last_normal_ts = df['timestamp_normal'].dropna().max()
    normal_interp = np.where(all_timestamps > last_normal_ts, min_power, normal_interp)
    max_power = max(buggy_interp.max(), normal_interp.max())

    figure, ax = plt.subplots(figsize=(7,2.5))
    ax.plot(all_timestamps, buggy_interp, label='dist.Join', color='#E06666', linewidth=2)
    ax.plot(all_timestamps, normal_interp, label='early exit', color='#3D85C6', linewidth=2)
    # ax.axvline(5.9, color='black', linestyle='--', linewidth=1.5)
    ax.annotate('Enter idle mode', xy=(5.8, 100), 
            xycoords='data', xytext=(0.5, 0.43), textcoords='axes fraction',
            arrowprops=dict(color='black', arrowstyle="->", connectionstyle="arc3"),
            horizontalalignment='right', verticalalignment='top',
            color='maroon',fontweight='bold', fontsize=12,ha='center',)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Power (W)', fontsize=12)
    ax.set_ylim(min_power * 0.95, max_power * 1.05)
    ax.legend()
    ax.grid(True, linestyle='--', lw=0.5)
    figure.tight_layout()
    if args.output:
        figure.savefig(args.output, bbox_inches='tight', pad_inches=0.05)
    plt.show()

if __name__ == '__main__':
    args = parser.parse_args()
    plot_power_compare(args)
