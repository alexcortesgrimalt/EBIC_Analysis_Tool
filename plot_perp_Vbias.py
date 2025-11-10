import argparse
import glob
import os
import re
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#!/usr/bin/env python3



SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PERP_DIR = os.path.join(SCRIPT_DIR, 'perpendicular_plots')
if os.path.isdir(PERP_DIR):
    try:
        os.chdir(PERP_DIR)
    except Exception:
        pass
FNAME_RE = re.compile(r'(?P<sample>[^_]+)_(?P<bias>-?\d+)_perp_(?P<num>\d+)\.csv$', re.IGNORECASE)


def parse_filename(fname):
    base = os.path.basename(fname)
    m = FNAME_RE.match(base)
    if not m:
        return None
    return m.groupdict()


def read_data_file(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    col_map = {}
    for c in df.columns:
        if 'dist' in c:
            col_map['distance'] = c
        if c == 'current' or 'curr' in c:
            col_map['current'] = c
        if c == 'sem' or 'std' in c:
            col_map['sem'] = c
    if 'distance' not in col_map or 'current' not in col_map:
        raise ValueError(f"File {path} missing required columns (distance,current)")
    df = df.rename(columns={col_map['distance']: 'distance', col_map['current']: 'current'})
    if 'sem' in col_map:
        df = df.rename(columns={col_map['sem']: 'sem'})
    df['distance'] = pd.to_numeric(df['distance'], errors='coerce')
    df['current'] = pd.to_numeric(df['current'], errors='coerce')
    if 'sem' in df.columns:
        df['sem'] = pd.to_numeric(df['sem'], errors='coerce')
    df = df.dropna(subset=['distance', 'current'])
    return df[['distance', 'current']].reset_index(drop=True)


def aggregate_group(dfs):
    concat = pd.concat(dfs, ignore_index=True)
    grp = concat.groupby('distance', as_index=False)['current'].agg(['mean', 'sem', 'count']).reset_index()
    grp = grp.rename(columns={'mean': 'current_mean', 'sem': 'current_sem', 'count': 'n'})
    return grp


def bias_to_volt(bias_str):
    try:
        v = int(bias_str) / 10.0
    except Exception:
        v = float(bias_str)
    return v


def plot_sample(sample, bias_aggregates, out_dir, show=False):
    # Linear current plot
    plt.figure(figsize=(7, 5))
    cmap = plt.get_cmap('tab10')
    for i, (bias_str, agg_df) in enumerate(sorted(bias_aggregates.items(), key=lambda x: int(x[0]))):
        if agg_df.empty:
            continue
        x = agg_df['distance'].values
        y = agg_df['current_mean'].values
        yerr = agg_df['current_sem'].values
        n = int(agg_df['n'].max())  # number of repeats contributing (approx)
        color = cmap(i % 10)
        bias_v = bias_to_volt(bias_str)
        label = f"{bias_v:.2f} V (n≈{n})"
        plt.plot(x, y, label=label, color=color)
        if not np.all(np.isnan(yerr)):
            low = y - yerr
            high = y + yerr
            plt.fill_between(x, low, high, color=color, alpha=0.25)
    plt.xlabel('Distance')
    plt.ylabel('Current (nA)')
    plt.title(f'{sample} — current vs distance (perp)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{sample}_perp_current_vs_distance.png")
    plt.savefig(out_path, dpi=200)
    if show:
        plt.show()
    plt.close()
    print(f"Saved: {out_path}")

    # Log10(|current|) plot
    plt.figure(figsize=(7, 5))
    for i, (bias_str, agg_df) in enumerate(sorted(bias_aggregates.items(), key=lambda x: int(x[0]))):
        if agg_df.empty:
            continue
        x = agg_df['distance'].values
        y_mean = agg_df['current_mean'].values
        y_sem = agg_df['current_sem'].values
        n = int(agg_df['n'].max())
        color = cmap(i % 10)
        bias_v = bias_to_volt(bias_str)
        label = f"{bias_v:.2f} V (n≈{n})"

        # Work with absolute values for the log. Avoid log(0) by clipping to a small floor.
        abs_y = np.abs(y_mean)
        floor = 1e-12
        abs_y_clipped = np.clip(abs_y, floor, None)
        y_log_raw = np.log10(abs_y_clipped)

        # Floor the log values at -1.0
        floor_log = -1.0
        y_log = np.where(y_log_raw < floor_log, floor_log, y_log_raw)

        # Propagate SEM to log space: sigma_log = sem / (|mean| * ln(10))
        if 'current_sem' in agg_df.columns and not np.all(np.isnan(y_sem)):
            denom = np.clip(abs_y, floor, None) * np.log(10)
            yerr_log_raw = y_sem / denom
            # If the raw log was floored, disable the error bar (set to 0) so it doesn't extend below the floor
            yerr_log = np.where(y_log_raw < floor_log, 0.0, yerr_log_raw)
        else:
            yerr_log = np.full_like(y_log, np.nan)

        plt.plot(x, y_log, label=label, color=color)
        if not np.all(np.isnan(yerr_log)):
            low = y_log - yerr_log
            high = y_log + yerr_log
            plt.fill_between(x, low, high, color=color, alpha=0.25)

    plt.xlabel('Distance')
    plt.ylabel('log10(|Current|)')
    plt.title(f'{sample} — log10(|current|) vs distance (perp)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    out_path_log = os.path.join(out_dir, f"{sample}_perp_log10_current_vs_distance.png")
    plt.savefig(out_path_log, dpi=200)
    if show:
        plt.show()
    plt.close()
    print(f"Saved: {out_path_log}")


def main(directory='.', pattern='*_perp_*.csv', out_dir=None, show=False):
    if out_dir is None:
        out_dir = directory
    os.makedirs(out_dir, exist_ok=True)

    files = glob.glob(os.path.join(directory, pattern))
    if not files:
        print("No files found matching pattern in directory:", directory)
        return

    data_map = defaultdict(lambda: defaultdict(list))
    for f in files:
        meta = parse_filename(f)
        if not meta:
            continue
        sample = meta['sample']
        bias = meta['bias']
        try:
            df = read_data_file(f)
        except Exception as e:
            print(f"Skipping {f}: {e}")
            continue
        data_map[sample][bias].append(df)

    if not data_map:
        print("No valid files parsed.")
        return

    for sample, biases in data_map.items():
        bias_aggregates = {}
        for bias_str, dfs in biases.items():
            agg = aggregate_group(dfs)
            bias_aggregates[bias_str] = agg
        plot_sample(sample, bias_aggregates, out_dir, show=show)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Plot perpendicular current vs distance for multiple biases and samples.")
    p.add_argument('directory', nargs='?', default='.', help='Directory to scan for CSV files')
    p.add_argument('--pattern', default='*_perp_*.csv', help='Glob pattern to find files')
    p.add_argument('--out', default=None, help='Output directory for PNGs (default: same as input directory)')
    p.add_argument('--show', action='store_true', help='Show plots interactively')
    args = p.parse_args()
    main(directory=args.directory, pattern=args.pattern, out_dir=args.out, show=args.show)