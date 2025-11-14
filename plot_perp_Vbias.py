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
# Newer filenames can contain many underscores and additional tokens.
# Capture everything up to "_perp_<num>" as a single prefix, then heuristically
# split that prefix into sample and optional bias token.
FNAME_RE = re.compile(r'(?P<prefix>.+)_perp_(?P<num>\d+)\.(?P<ext>csv|png)$', re.IGNORECASE)


def parse_filename(fname):
    """Parse filename and return dict with keys: sample, bias, num, ext.

    Heuristics:
    - Match <prefix>_perp_<num>.<ext>
    - Attempt to treat the last underscore-separated token of <prefix> as a bias
      if it looks like an integer (e.g. "-02") or common markers like "noVb".
    - Otherwise the full prefix is returned as `sample` and `bias` is set to None.
    """
    base = os.path.basename(fname)
    m = FNAME_RE.match(base)
    if not m:
        return None
    gd = m.groupdict()
    prefix = gd.get('prefix')
    num = gd.get('num')
    ext = gd.get('ext')

    # Split prefix into tokens and try to detect a bias token at the end
    toks = prefix.split('_') if prefix else [prefix]
    bias = None
    sample = prefix
    last = toks[-1] if toks else ''

    # Recognize bias indicators in several common formats:
    # - simple integer token (e.g. '-02') -> legacy style
    # - tokens like 'Vb-01' or 'Vb01' -> extract numeric part
    # - 'noVb' (case-insensitive) -> treat as zero-bias marker
    if re.match(r'^-?\d+$', last):
        bias = last
        sample = '_'.join(toks[:-1]) if len(toks) > 1 else toks[0]
    else:
        m_vb = re.match(r'^[Vv][bB](-?\d+)$', last)
        if m_vb:
            # store numeric portion (e.g. '-01' or '01') so downstream code treats it like '-01'
            bias = m_vb.group(1)
            sample = '_'.join(toks[:-1]) if len(toks) > 1 else toks[0]
        elif re.match(r'^(?:noVb|novb)$', last, re.IGNORECASE):
            bias = last
            sample = '_'.join(toks[:-1]) if len(toks) > 1 else toks[0]

    return {'sample': sample, 'bias': bias, 'num': num, 'ext': ext}


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
    """Convert bias token into a voltage in volts when possible.

    Handles tokens like '-02' (interpreted as -0.2 V if original code used /10),
    numeric strings, and markers like 'noVb'. Returns None when conversion is not possible.
    """
    if bias_str is None:
        return None
    s = str(bias_str).strip()
    if s == '':
        return None
    # treat noVb / novb as explicit zero
    if re.search(r'novb', s, re.IGNORECASE):
        return 0.0

    # If token contains a decimal point, parse as float
    if '.' in s:
        try:
            return float(s)
        except Exception:
            return None

    # Signed numeric string (e.g. '-02', '01', '1')
    m = re.match(r'^(-?)(\d+)$', s)
    if m:
        sign = -1.0 if m.group(1) == '-' else 1.0
        digits = m.group(2)
        # Heuristic: if digits start with a '0' (leading zero) or have length >= 2,
        # apply legacy behavior (divide by 10) to match examples like '01' -> 0.1
        # Otherwise (single non-zero digit) treat as whole volts (e.g. '1' -> 1.0)
        if digits.startswith('0') or len(digits) >= 2:
            try:
                return sign * (int(digits) / 10.0)
            except Exception:
                return None
        else:
            try:
                return sign * float(int(digits))
            except Exception:
                return None

    # Last resort: try to extract a numeric substring
    m2 = re.search(r'(-?\d+)', s)
    if m2:
        try:
            return int(m2.group(1)) / 10.0
        except Exception:
            return None
    return None


def _bias_sort_key(bias_str):
    """Return a numeric sort key for a bias token using volts; non-numeric values -> 0."""
    try:
        v = bias_to_volt(bias_str)
        return 0.0 if v is None else float(v)
    except Exception:
        return 0.0


def plot_sample(sample, bias_aggregates, out_dir, show=False, suffix=''):
    # Linear current plot
    plt.figure(figsize=(7, 5))
    cmap = plt.get_cmap('tab10')
    for i, (bias_str, agg_df) in enumerate(sorted(bias_aggregates.items(), key=lambda x: _bias_sort_key(x[0]))):
        if agg_df.empty:
            continue
        x = agg_df['distance'].values
        y = agg_df['current_mean'].values
        yerr = agg_df['current_sem'].values
        n = int(agg_df['n'].max())  # number of repeats contributing (approx)
        color = cmap(i % 10)
        bias_v = bias_to_volt(bias_str)
        if bias_v is None:
            label = f"{bias_str} (n≈{n})"
        else:
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
    name_base = f"{sample}{suffix}_perp_current_vs_distance.png"
    out_path = os.path.join(out_dir, name_base)
    plt.savefig(out_path, dpi=200)
    if show:
        plt.show()
    plt.close()
    print(f"Saved: {out_path}")

    # Log10(|current|) plot
    plt.figure(figsize=(7, 5))
    for i, (bias_str, agg_df) in enumerate(sorted(bias_aggregates.items(), key=lambda x: _bias_sort_key(x[0]))):
        if agg_df.empty:
            continue
        x = agg_df['distance'].values
        y_mean = agg_df['current_mean'].values
        y_sem = agg_df['current_sem'].values
        n = int(agg_df['n'].max())
        color = cmap(i % 10)
        bias_v = bias_to_volt(bias_str)
        if bias_v is None:
            label = f"{bias_str} (n≈{n})"
        else:
            label = f"{bias_v:.2f} V (n≈{n})"

        # Work with absolute values for the natural log. Avoid log(0) by clipping to a small floor.
        abs_y = np.abs(y_mean)
        floor = 1e-12
        abs_y_clipped = np.clip(abs_y, floor, None)
        # Natural log
        y_log_raw = np.log(abs_y_clipped)

        # Floor the ln values at ln(floor) to avoid huge negatives
        floor_log = np.log(floor)
        y_log = np.where(y_log_raw < floor_log, floor_log, y_log_raw)

        if 'current_sem' in agg_df.columns and not np.all(np.isnan(y_sem)):
            denom = np.clip(abs_y, floor, None)
            yerr_log_raw = y_sem / denom
            yerr_log = np.where(y_log_raw < floor_log, 0.0, yerr_log_raw)
        else:
            yerr_log = np.full_like(y_log, np.nan)

        plt.plot(x, y_log, label=label, color=color)
        if not np.all(np.isnan(yerr_log)):
            low = y_log - yerr_log
            high = y_log + yerr_log
            plt.fill_between(x, low, high, color=color, alpha=0.25)

    plt.xlabel('Distance')
    plt.ylabel('ln(|Current|)')
    plt.title(f'{sample} — ln(|current|) vs distance (perp)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    out_path_log = os.path.join(out_dir, f"{sample}{suffix}_perp_ln_current_vs_distance.png")
    plt.savefig(out_path_log, dpi=200)
    if show:
        plt.show()
    plt.close()
    print(f"Saved: {out_path_log}")


def main(directory='.', pattern='*_perp_*.csv', out_dir=None, show=False):
    # Default output directory: create a dedicated folder 'output_perp_vbias'
    if out_dir is None:
        out_dir = os.path.join(directory, 'output_perp_vbias')
    os.makedirs(out_dir, exist_ok=True)

    files = glob.glob(os.path.join(directory, pattern))
    if not files:
        print("No files found matching pattern in directory:", directory)
        return

    data_map = defaultdict(lambda: defaultdict(list))
    for f in files:
        meta = parse_filename(f)
        if not meta:
            print(f"Skipping (unrecognized name): {f}")
            continue

        # If the matched file is not a CSV (e.g. PNG), try to find a corresponding CSV
        # with the same prefix. Otherwise skip non-CSV files because this script reads CSV data.
        ext = meta.get('ext', '').lower()
        csv_path = f
        if ext != 'csv':
            alt = os.path.splitext(f)[0] + '.csv'
            if os.path.exists(alt):
                csv_path = alt
                meta = parse_filename(alt) or meta
            else:
                print(f"Skipping non-CSV file (no matching CSV found): {f}")
                continue

        sample = meta.get('sample')
        bias = meta.get('bias')
        try:
            df = read_data_file(csv_path)
        except Exception as e:
            print(f"Skipping {csv_path}: {e}")
            continue
        data_map[sample][bias].append(df)

    # No interactive prompt: produce two sets of plots per sample
    # - one including all biases (with negatives)
    # - one excluding negative biases (without negatives)

    if not data_map:
        print("No valid files parsed.")
        return

    for sample, biases in data_map.items():
        # Full set (all biases)
        full_aggregates = {b: aggregate_group(dfs) for b, dfs in biases.items()}

        # Filtered set (exclude negative numeric voltages)
        no_neg_aggregates = {}
        for b, dfs in biases.items():
            v = bias_to_volt(b)
            if v is not None and v < 0:
                continue
            no_neg_aggregates[b] = aggregate_group(dfs)

        # Save both plots: with negatives and without negatives (if different)
        if full_aggregates:
            plot_sample(sample, full_aggregates, out_dir, show=show, suffix='_with_neg')
        if no_neg_aggregates:
            # Avoid duplicating identical plots
            if set(no_neg_aggregates.keys()) != set(full_aggregates.keys()):
                plot_sample(sample, no_neg_aggregates, out_dir, show=show, suffix='_without_neg')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Plot perpendicular current vs distance for multiple biases and samples.")
    p.add_argument('directory', nargs='?', default='.', help='Directory to scan for CSV files')
    p.add_argument('--pattern', default='*_perp_*.csv', help='Glob pattern to find files')
    p.add_argument('--out', default=None, help='Output directory for PNGs (default: same as input directory)')
    p.add_argument('--show', action='store_true', help='Show plots interactively')
    p.add_argument('--include-negative', action='store_true', help='Include negative biases without prompting')
    args = p.parse_args()
    # Pass include_negative to main via environment prompt-handling: if user passed
    # --include-negative we will set the flag and avoid prompting inside main.
    if args.include_negative:
        # Monkey-patch input to auto-yes for negatives inside main
        try:
            _orig_input = __builtins__['input'] if isinstance(__builtins__, dict) else __builtins__.input
        except Exception:
            _orig_input = input
        def _auto_yes(prompt=''):
            print(prompt + ' [auto-yes]')
            return 'y'
        try:
            if isinstance(__builtins__, dict):
                __builtins__['input'] = _auto_yes
            else:
                __builtins__.input = _auto_yes
        except Exception:
            pass

    try:
        main(directory=args.directory, pattern=args.pattern, out_dir=args.out, show=args.show)
    finally:
        # restore input if we patched it
        try:
            if args.include_negative:
                if isinstance(__builtins__, dict):
                    __builtins__['input'] = _orig_input
                else:
                    __builtins__.input = _orig_input
        except Exception:
            pass