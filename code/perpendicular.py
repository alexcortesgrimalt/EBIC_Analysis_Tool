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
                                     source_name=None):
    """
    Calculate perpendicular profiles along a main line.
    Returns a list of dicts with SEM, current, and intersection info.
    Parameters
    ----------
    source_name : str or None
        Optional name or path of the source file the data was taken from.
        When provided, it will be attached to each returned profile as
        the 'source_name' key so plotting/export routines can use it.
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
    length_px = length_um / (pixel_size_m * 1e6)

    distances = np.linspace(0, len(dense_line)-1, num_lines).astype(int)
    profiles = []

    for i, idx in enumerate(distances):
        center = dense_line[idx]
        p_start = center - perp_unit*(length_px/2)
        p_end = center + perp_unit*(length_px/2)
        x = np.linspace(p_start[0], p_end[0], int(length_px))
        y = np.linspace(p_start[1], p_end[1], int(length_px))
        x = np.clip(x, 0, sem_data.shape[1]-1)
        y = np.clip(y, 0, sem_data.shape[0]-1)

        sem_values = map_coordinates(sem_data, [y, x], order=1, mode='nearest')
        current_values = map_coordinates(current_data, [y, x], order=1, mode='nearest')
        dist_um_line = np.linspace(-length_um/2, length_um/2, len(sem_values))

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
        # middle = log10(Current), bottom = derivatives (dI/dx and dlnI/dx)
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

        # Prepare safe current for plotting and set log y-scale on middle axis
        pos = cur[cur > 0]
        if pos.size > 0:
            floor = max(np.min(pos) * 0.1, 1e-12)
        else:
            floor = 1e-12
        cur_safe = np.maximum(cur, floor)
        # Debug: floor used for log-safe current
        if debug:
            print(f"[perp] log-floor used = {floor:.3e} (pos_count={pos.size})")

        # Plot raw current values but use a logarithmic y-axis for the middle subplot
        ax_log.plot(dist_um, cur_safe, color='tab:orange', linewidth=1.5, label='Current (nA)')
        ax_log.set_yscale('log')

        # Mark intersection point on all axes if available
        if prof.get("intersection_idx") is not None:
            idx = prof["intersection_idx"]
            ax1.scatter(dist_um[idx], sem_vals_norm[idx], color='lime', s=50, marker='x', zorder=10)
            ax1b.scatter(dist_um[idx], cur[idx], color='lime', s=50, marker='x', zorder=10)
            ax_log.scatter(dist_um[idx], cur_safe[idx], color='lime', s=50, marker='x', zorder=10)
            # mark on derivative axis at x position (y position will be set by plot)
            ax_der.scatter(dist_um[idx], 0, color='lime', s=50, marker='x', zorder=10)

        # include source name in subplot title for clarity
        if prof.get('source_name'):
            ax1.set_title(f"{os.path.splitext(os.path.basename(str(prof.get('source_name'))))[0]} - Perpendicular {prof['id']+1}")
        else:
            ax1.set_title(f"Perpendicular {prof['id']+1}")

        ax1.set_ylabel("SEM Contrast (norm)", color='tab:blue')
        ax1b.set_ylabel("Current (nA)", color='tab:red')
        ax_log.set_xlabel("Distance (µm)")
        ax_log.set_ylabel("Current (nA) [log scale]", color='tab:orange')
        ax_der.set_xlabel("Distance (µm)")
        ax_der.set_ylabel("Derivatives", color='tab:green')

        # Combine legends from ax1 and ax1b
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1b.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        ax_log.legend(loc='upper left')

        # Compute measured derivatives (from raw profile)
        # Using windowed gradient for smoother derivatives
        x_vals = np.array(dist_um, dtype=float)
        gradient_window = 9  # window size for gradient computation (must be odd)
        if x_vals.size > 1:
            try:
                dI_dx_meas = gradient_with_window(x_vals, cur, window=gradient_window)
            except Exception:
                dI_dx_meas = np.zeros_like(cur)
            try:
                dlnI_dx_meas = gradient_with_window(x_vals, np.log(cur_safe), window=gradient_window)
            except Exception:
                dlnI_dx_meas = np.zeros_like(cur)
        else:
            dI_dx_meas = np.zeros_like(cur)
            dlnI_dx_meas = np.zeros_like(cur)

        # By default plot the measured derivatives, but if a fit is attached to
        # this profile and contains precomputed fit derivatives, prefer the
        # derivative of the interpolated/fitted curve. We will interpolate the
        # fit-derived derivatives onto the profile x-axis so overlays match.
        dI_dx_plot = dI_dx_meas
        dlnI_dx_plot = dlnI_dx_meas
        fit_dI_dx_interp = None
        fit_dlnI_dx_interp = None

        # Collect candidate fit dicts from common keys
        fit_candidates = []
        # single fit dict
        if isinstance(prof.get('fit'), dict):
            fit_candidates.append(prof.get('fit'))
        # list of fits (fit_sides style)
        if isinstance(prof.get('fit'), list):
            fit_candidates.extend(prof.get('fit'))
        if isinstance(prof.get('fit_sides'), list):
            fit_candidates.extend(prof.get('fit_sides'))
        # depletion/best fit containers
        if isinstance(prof.get('depletion'), dict):
            best_left = prof['depletion'].get('best_left_fit')
            best_right = prof['depletion'].get('best_right_fit')
            if isinstance(best_left, dict):
                fit_candidates.append(best_left)
            if isinstance(best_right, dict):
                fit_candidates.append(best_right)

        # Choose best fit candidate (highest R² if present)
        best_fit = None
        if fit_candidates:
            scored = [(f.get('r2', -np.inf), f) for f in fit_candidates if isinstance(f, dict)]
            if scored:
                best_fit = max(scored, key=lambda t: t[0])[1]

        # If we have a best fit, try to obtain fit-derived derivatives
        if best_fit is not None:
            try:
                # prefer precomputed fit derivative arrays if present
                fit_x = None
                for k in ('global_x_vals', 'global_x', 'x_vals', 'x'):
                    if k in best_fit:
                        fit_x = np.array(best_fit[k], dtype=float)
                        break

                # pick the fit derivative arrays or compute them from fit_curve
                # Use windowed gradient for smoother derivatives
                if 'fit_dI_dx' in best_fit:
                    fit_dI = np.array(best_fit['fit_dI_dx'], dtype=float)
                else:
                    fit_curve_arr = np.array(best_fit.get('fit_curve', []), dtype=float)
                    if fit_x is not None and fit_x.size > 1:
                        try:
                            fit_dI = gradient_with_window(fit_x, fit_curve_arr, window=gradient_window)
                        except Exception:
                            fit_dI = None
                    else:
                        fit_dI = None

                if 'fit_dlnI_dx' in best_fit:
                    fit_dln = np.array(best_fit['fit_dlnI_dx'], dtype=float)
                else:
                    fit_curve_arr = np.array(best_fit.get('fit_curve', []), dtype=float)
                    if fit_x is not None and fit_x.size > 1:
                        try:
                            fit_dln = gradient_with_window(fit_x, np.log(np.maximum(fit_curve_arr, 1e-12)), window=gradient_window)
                        except Exception:
                            fit_dln = None
                    else:
                        fit_dln = None

                # Interpolate fit-derived derivatives onto profile x axis if possible
                if fit_x is not None and fit_x.size > 1 and fit_dI is not None:
                    # ensure fit_x sorted
                    sort_idx = np.argsort(fit_x)
                    fx_sorted = fit_x[sort_idx]
                    dI_sorted = fit_dI[sort_idx]
                    fit_dI_dx_interp = np.interp(x_vals, fx_sorted, dI_sorted, left=np.nan, right=np.nan)
                    dI_dx_plot = fit_dI_dx_interp
                    fit_dI_dx_interp = fit_dI_dx_interp
                if fit_x is not None and fit_x.size > 1 and fit_dln is not None:
                    sort_idx = np.argsort(fit_x)
                    fx_sorted = fit_x[sort_idx]
                    dln_sorted = fit_dln[sort_idx]
                    fit_dlnI_dx_interp = np.interp(x_vals, fx_sorted, dln_sorted, left=np.nan, right=np.nan)
                    dlnI_dx_plot = fit_dlnI_dx_interp
            except Exception:
                # fallback to measured derivatives
                dI_dx_plot = dI_dx_meas
                dlnI_dx_plot = dlnI_dx_meas

        # Debug: derivatives stats (min/max and NaN check)
        try:
            print(f"[perp] dI/dx (plotted): min={np.nanmin(dI_dx_plot):.3e} max={np.nanmax(dI_dx_plot):.3e} n_nan={np.isnan(dI_dx_plot).sum()}")
            print(f"[perp] dlnI/dx (plotted): min={np.nanmin(dlnI_dx_plot):.3e} max={np.nanmax(dlnI_dx_plot):.3e} n_nan={np.isnan(dlnI_dx_plot).sum()}")
        except Exception:
            print("[perp] derivative stats unavailable")

        # Plot derivatives: prefer fit-derived overlay when available (solid),
        # and also optionally show measured derivatives as dotted (for comparison)
        label_dI = 'dI/dx (nA/µm)'
        label_dln = 'dlnI/dx (1/µm)'
        if fit_dI_dx_interp is not None:
            ax_der.plot(dist_um, dI_dx_plot, color='tab:purple', linewidth=1.6, label=label_dI + ' (fit)')
            # also show measured as dotted for reference
            ax_der.plot(dist_um, dI_dx_meas, color='tab:purple', linewidth=0.8, linestyle=':', label='dI/dx (meas)')
        else:
            ax_der.plot(dist_um, dI_dx_plot, color='tab:purple', linewidth=1.2, label=label_dI)

        if fit_dlnI_dx_interp is not None:
            ax_der.plot(dist_um, dlnI_dx_plot, color='tab:green', linewidth=1.4, label=label_dln + ' (fit)')
            ax_der.plot(dist_um, dlnI_dx_meas, color='tab:green', linewidth=0.8, linestyle=':', label='dlnI/dx (meas)')
        else:
            ax_der.plot(dist_um, dlnI_dx_plot, color='tab:green', linewidth=1.0, label=label_dln)
        ax_der.legend(loc='upper left')

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
            # Also export numeric profile data as CSV (dist_um, sem, current, dI_dx, dlnI_dx)
            try:
                csv_path = os.path.join(out_dir, f'{base_name}_perp_{prof["id"]+1:02d}.csv')
                # Ensure derivatives are available for saving
                x_vals = np.array(dist_um, dtype=float)
                cur_arr = np.array(cur_vals, dtype=float)
                if x_vals.size > 1:
                    dI_dx_save = np.gradient(cur_arr, x_vals)
                    pos = cur_arr[cur_arr > 0]
                    if pos.size > 0:
                        floor_save = max(np.min(pos) * 0.1, 1e-12)
                    else:
                        floor_save = 1e-12
                    cur_safe_save = np.maximum(cur_arr, floor_save)
                    dlnI_dx_save = np.gradient(np.log(cur_safe_save), x_vals)
                else:
                    dI_dx_save = np.zeros_like(cur_arr)
                    dlnI_dx_save = np.zeros_like(cur_arr)

                # If fit-derived interpolated derivatives are available, include them
                if fit_dI_dx_interp is not None or fit_dlnI_dx_interp is not None:
                    # make arrays same length as x_vals (they already are interpolated)
                    fdI = fit_dI_dx_interp if fit_dI_dx_interp is not None else np.full_like(dI_dx_save, np.nan)
                    fdln = fit_dlnI_dx_interp if fit_dlnI_dx_interp is not None else np.full_like(dlnI_dx_save, np.nan)
                    data = np.column_stack([dist_um, sem_vals, cur_vals, dI_dx_save, dlnI_dx_save, fdI, fdln])
                    header = 'dist_um,sem,current,dI_dx,dlnI_dx,fit_dI_dx,fit_dlnI_dx'
                else:
                    data = np.column_stack([dist_um, sem_vals, cur_vals, dI_dx_save, dlnI_dx_save])
                    header = 'dist_um,sem,current,dI_dx,dlnI_dx'
                # Use scientific notation with reasonable precision
                np.savetxt(csv_path, data, delimiter=',', header=header, comments='', fmt='%.6e')
                print(f"Saved perpendicular CSV to {csv_path}")
            except Exception as e_csv:
                print("Failed to save perpendicular CSV:", e_csv)
        except Exception as e:
            print("Failed to save perpendicular plot:", e)
