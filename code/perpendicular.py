# perpendicular.py
import numpy as np
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt


def gradient_with_window(x, y, window=9):
    """
    Compute gradient using a moving window with linear fit.
    This provides smoother derivatives than np.gradient by fitting
    a line to a local window around each point.
    
    Parameters
    ----------
    x : array-like
        x coordinates (must be 1D)
    y : array-like
        y values (must be 1D, same length as x)
    window : int, optional
        Window size (must be odd). Larger values give smoother derivatives.
        Default is 7.
    
    Returns
    -------
    dy : ndarray
        Derivative dy/dx computed via local linear fits
    """
    if window % 2 == 0:
        raise ValueError("window must be odd")
    n = len(x)
    k = window // 2
    dy = np.full(n, np.nan)
    # ensure x is numpy array
    x_arr = x.to_numpy() if hasattr(x, "to_numpy") else np.asarray(x)
    y_arr = y if isinstance(y, np.ndarray) else np.asarray(y)
    for i in range(n):
        lo = max(0, i - k)
        hi = min(n, i + k + 1)
        xi = x_arr[lo:hi]
        yi = y_arr[lo:hi]
        if xi.size >= 2:
            # linear fit: slope is derivative
            p = np.polyfit(xi, yi, 1)
            dy[i] = p[0]
    return dy


def calculate_perpendicular_profiles(line_coords, num_lines, length_um,
                                     sem_data, current_data,
                                     pixel_size_m=1e-6, detected_junction=None,
                                     source_name=None, length_left_um=None, length_right_um=None):
    """
    Calculate perpendicular profiles along a main line.
    Returns a list of dicts with SEM, current, and intersection info.
    Parameters
    ----------
    source_name : str or None
        Optional name or path of the source file the data was taken from.
        When provided, it will be attached to each returned profile as
        the 'source_name' key so plotting/export routines can use it.
    length_left_um : float or None
        Length extending to the left side of the main line (µm).
        If None, uses length_um/2 for symmetric profiles.
    length_right_um : float or None
        Length extending to the right side of the main line (µm).
        If None, uses length_um/2 for symmetric profiles.
    """
    # Create dense line points
    if isinstance(line_coords, np.ndarray) and line_coords.shape[1] == 2:
        x0, y0 = line_coords[0]
        x1, y1 = line_coords[-1]
        dense_line = line_coords
    else:
        (x0, y0), (x1, y1) = line_coords
        line_length = int(np.ceil(np.hypot(x1 - x0, y1 - y0)))
        xs = np.linspace(x0, x1, line_length)
        ys = np.linspace(y0, y1, line_length)
        dense_line = np.column_stack([xs, ys])

    # Main & perpendicular vectors
    main_vec = np.array([x1 - x0, y1 - y0], dtype=float)
    main_len = np.linalg.norm(main_vec)
    if main_len == 0: return []

    main_unit = main_vec / main_len
    perp_unit = np.array([-main_unit[1], main_unit[0]])
    
    # Use asymmetric lengths if provided, otherwise symmetric
    if length_left_um is None:
        length_left_um = length_um / 2
    if length_right_um is None:
        length_right_um = length_um / 2
    
    length_left_px = length_left_um / (pixel_size_m * 1e6)
    length_right_px = length_right_um / (pixel_size_m * 1e6)
    total_length_px = length_left_px + length_right_px

    distances = np.linspace(0, len(dense_line)-1, num_lines).astype(int)
    profiles = []

    for i, idx in enumerate(distances):
        center = dense_line[idx]
        p_start = center - perp_unit * length_left_px
        p_end = center + perp_unit * length_right_px
        x = np.linspace(p_start[0], p_end[0], int(total_length_px))
        y = np.linspace(p_start[1], p_end[1], int(total_length_px))
        x = np.clip(x, 0, sem_data.shape[1]-1)
        y = np.clip(y, 0, sem_data.shape[0]-1)

        sem_values = map_coordinates(sem_data, [y, x], order=1, mode='nearest')
        current_values = map_coordinates(current_data, [y, x], order=1, mode='nearest')
        dist_um_line = np.linspace(-length_left_um, length_right_um, len(sem_values))

        # Intersection with junction if available
        intersection, intersection_idx = None, None
        if detected_junction is not None:
            line_vec = p_end - p_start
            line_len2 = np.sum(line_vec**2)
            t = np.dot(detected_junction - p_start, line_vec) / line_len2
            t = np.clip(t, 0, 1)
            proj = p_start + np.outer(t, line_vec)
            distances_to_proj = np.linalg.norm(detected_junction - proj, axis=1)
            idx_min = np.argmin(distances_to_proj)
            intersection = proj[idx_min]
            intersection_idx = int(round(t[idx_min]*(len(sem_values)-1)))
            intersection_idx = np.clip(intersection_idx, 0, len(sem_values)-1)

        profiles.append({
            "id": i,
            "dist_um": dist_um_line,
            "sem": sem_values,
            "current": current_values,
            "line_coords": (p_start, p_end),
            "intersection": intersection,
            "intersection_idx": intersection_idx
        })
        # attach source name so callers (plotting/export) can find the
        # analyzed filename automatically
        if source_name is not None:
            profiles[-1]['source_name'] = source_name

    return profiles


def plot_perpendicular_profiles(profiles, ax=None, fig=None, source_name=None, debug=False):
    """
    Plot perpendicular lines and scrollable profile plots.
    
    Parameters
    ----------
    debug : bool
        If True, print debug information during plotting
    """
    if not profiles: return
    if ax is not None and fig is not None:
        for prof in profiles:
            p_start, p_end = prof['line_coords']
            ax.plot([p_start[0], p_end[0]], [p_start[1], p_end[1]], 'r--', linewidth=1)
            if prof.get('intersection') is not None:
                inter = prof['intersection']
                ax.scatter(inter[0], inter[1], color='lime', s=50, marker='x', zorder=20)
        try:
            fig.canvas.draw_idle()
            # force immediate draw in some backends where idle draw doesn't update
            fig.canvas.draw()
        except Exception:
            pass

    # Plot each perpendicular profile
    import tkinter as tk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import os
    win = tk.Toplevel()
    # Display source name in window title if available
    display_name = source_name
    if not display_name and profiles and isinstance(profiles, list) and profiles[0].get('source_name'):
        display_name = profiles[0].get('source_name')
    if display_name:
        win.title(f"Perpendicular Profiles - {os.path.splitext(os.path.basename(str(display_name)))[0]}")
    else:
        win.title("Perpendicular Profiles")
    win.geometry("1200x800")
    canvas = tk.Canvas(win); scrollbar = tk.Scrollbar(win, orient="vertical", command=canvas.yview)
    scroll_frame = tk.Frame(canvas)
    scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0,0), window=scroll_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    for prof in profiles:
        dist_um = prof["dist_um"]
        sem_vals = prof["sem"]
        cur_vals = prof["current"]
        sem_vals_norm = (sem_vals - np.min(sem_vals)) / (np.ptp(sem_vals)+1e-12)
        # Debug print: basic profile info
        if debug:
            try:
                print(f"[perp] profile id={prof.get('id')} n_points={len(dist_um)}")
            except Exception:
                print("[perp] profile (id unknown) starting plot")
        # Create a 3-row plot: top = SEM (normalized) + Current (linear) via twin y,
        # middle = ln(Current), bottom = ln derivative (dlnI/dx only)
        fig_prof, (ax1, ax_log, ax_der) = plt.subplots(3, 1, sharex=True, figsize=(6, 7), gridspec_kw={'height_ratios': [1, 1, 0.8]})
        ax1.plot(dist_um, sem_vals_norm, color='tab:blue', linewidth=2, label='SEM (norm)')

        # Plot linear current on a twin axis above (original behavior)
        ax1b = ax1.twinx()
        cur = np.array(cur_vals)
        ax1b.plot(dist_um, cur, color='tab:red', linewidth=1.2, label='Current (nA)')
        # Debug: current stats
        if debug:
            try:
                print(f"[perp] current: min={np.nanmin(cur):.3e} max={np.nanmax(cur):.3e} mean={np.nanmean(cur):.3e}")
            except Exception:
                print("[perp] current: could not compute stats")

        # Prepare safe current for ln plotting
        pos = cur[cur > 0]
        if pos.size > 0:
            floor = max(np.min(pos) * 0.1, 1e-12)
        else:
            floor = 1e-12
        cur_safe = np.maximum(cur, floor)
        # Debug: floor used for log-safe current
        if debug:
            print(f"[perp] log-floor used = {floor:.3e} (pos_count={pos.size})")

        # Plot ln(current) on the middle subplot
        ln_current = np.log(cur_safe)
        ax_log.plot(dist_um, ln_current, color='tab:orange', linewidth=1.5, label='ln(Current)')

        # Mark intersection point on all axes if available
        if prof.get("intersection_idx") is not None:
            idx = prof["intersection_idx"]
            ax1.scatter(dist_um[idx], sem_vals_norm[idx], color='lime', s=50, marker='x', zorder=10)
            ax1b.scatter(dist_um[idx], cur[idx], color='lime', s=50, marker='x', zorder=10)
            ax_log.scatter(dist_um[idx], ln_current[idx], color='lime', s=50, marker='x', zorder=10)
            # mark on derivative axis at x position
            ax_der.axvline(dist_um[idx], color='lime', linestyle='--', alpha=0.5, zorder=10)

        # include source name in subplot title for clarity
        if prof.get('source_name'):
            ax1.set_title(f"{os.path.splitext(os.path.basename(str(prof.get('source_name'))))[0]} - Perpendicular {prof['id']+1}")
        else:
            ax1.set_title(f"Perpendicular {prof['id']+1}")

        ax1.set_ylabel("SEM Contrast (norm)", color='tab:blue')
        ax1b.set_ylabel("Current (nA)", color='tab:red')
        ax_log.set_ylabel("ln(Current)", color='tab:orange')
        ax_der.set_xlabel("Distance (µm)")
        ax_der.set_ylabel("dlnI/dx (1/µm)", color='tab:green')

        # Combine legends from ax1 and ax1b
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1b.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        ax_log.legend(loc='upper left')

        # Compute ln derivative (dlnI/dx) using windowed gradient
        x_vals = np.array(dist_um, dtype=float)
        gradient_window = 9
        if x_vals.size > 1:
            try:
                dlnI_dx = gradient_with_window(x_vals, np.log(cur_safe), window=gradient_window)
            except Exception:
                dlnI_dx = np.zeros_like(cur_safe)
        else:
            dlnI_dx = np.zeros_like(cur_safe)

        # Plot ln derivative only (no dI/dx)
        ax_der.plot(dist_um, dlnI_dx, color='tab:green', linewidth=1.2, label='dlnI/dx (1/µm)')
        ax_der.axhline(0, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
        ax_der.legend(loc='upper left')
        ax_der.grid(True, linestyle='--', alpha=0.3)

        fig_prof.tight_layout()
        canvas_plot = FigureCanvasTkAgg(fig_prof, master=scroll_frame)
        try:
            canvas_plot.draw()
        except Exception:
            pass
        widget = canvas_plot.get_tk_widget()
        widget.pack(pady=10)

        # Also save the figure to disk as a fallback so you always get the log plot
        try:
            out_dir = os.path.join(os.getcwd(), 'perpendicular_plots')
            os.makedirs(out_dir, exist_ok=True)

            # Determine base name for saved files. Prefer explicit source_name
            # argument, otherwise look in the profile dict for common keys.
            base_name = None
            if source_name:
                base_name = source_name
            else:
                # try common profile keys
                for k in ('source_name', 'file_name', 'filename'):
                    if profiles and isinstance(profiles, list) and profiles[0].get(k):
                        base_name = profiles[0].get(k)
                        break
            if base_name:
                # use only the file stem and sanitize
                base_name = os.path.splitext(os.path.basename(str(base_name)))[0]
                # replace spaces with underscores and remove path chars
                import re
                base_name = re.sub(r'[^A-Za-z0-9._-]', '_', base_name)
            else:
                base_name = 'perp'

            out_path = os.path.join(out_dir, f'{base_name}_perp_{prof["id"]+1:02d}.png')
            fig_prof.savefig(out_path, dpi=200, bbox_inches='tight')
            print(f"Saved perpendicular plot to {out_path}")
            # Export numeric profile data as CSV (dist_um, sem, current, dlnI_dx)
            try:
                csv_path = os.path.join(out_dir, f'{base_name}_perp_{prof["id"]+1:02d}.csv')
                # Compute dlnI/dx for CSV export
                x_vals_csv = np.array(dist_um, dtype=float)
                if x_vals_csv.size > 1:
                    try:
                        dlnI_dx_csv = gradient_with_window(x_vals_csv, np.log(cur_safe), window=9)
                    except Exception:
                        dlnI_dx_csv = np.zeros_like(cur_safe)
                else:
                    dlnI_dx_csv = np.zeros_like(cur_safe)
                data = np.column_stack([dist_um, sem_vals, cur_vals, dlnI_dx_csv])
                header = 'dist_um,sem,current,dlnI_dx'
                # Use scientific notation with reasonable precision
                np.savetxt(csv_path, data, delimiter=',', header=header, comments='', fmt='%.6e')
                print(f"Saved perpendicular CSV to {csv_path}")
            except Exception as e_csv:
                print("Failed to save perpendicular CSV:", e_csv)
        except Exception as e:
            print("Failed to save perpendicular plot:", e)
