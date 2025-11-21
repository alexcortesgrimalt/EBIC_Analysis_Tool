"""
GUI helper functions for SEM Viewer.
Provides dialogs, plotting utilities, and analysis workflows.
"""

import os
import tkinter as tk
from tkinter import simpledialog
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from scipy.ndimage import map_coordinates

from .DiffLenExt import DiffusionLengthExtractor
from .Junction_Analyser import JunctionAnalyzer
from .ROI_extractor import extract_line_rectangle
from .perpendicular import calculate_perpendicular_profiles, plot_perpendicular_profiles, gradient_with_window


def _extract_bias_voltage(sample_name):
    """
    Extract bias voltage from sample name.
    
    Rules:
    - 'noVb' -> 0
    - 'Vb0X' (e.g., Vb01, Vb02) -> 0.X (e.g., 0.1, 0.2)
    - 'VbX' (e.g., Vb1, Vb2) -> X (e.g., 1, 2)
    - '0XVb' (e.g., 01Vb, 02Vb) -> 0.X (e.g., 0.1, 0.2)
    
    Args:
        sample_name: Sample name string
        
    Returns:
        float: Bias voltage value, or None if not found
    """
    import re
    
    if not sample_name:
        return None
    
    name_lower = sample_name.lower()
    
    # Check for 'noVb' or 'novb'
    if 'novb' in name_lower:
        return 0.0
    
    # Pattern 1: Vb followed by digits (e.g., Vb01, Vb1, Vb2)
    match = re.search(r'vb0*(\d+)', name_lower)
    if match:
        value_str = match.group(1)
        value = int(value_str)
        # If original had leading zero (Vb01, Vb02), treat as decimal
        if match.group(0).startswith('vb0') and len(value_str) >= 1:
            return value / 10.0
        else:
            return float(value)
    
    # Pattern 2: Digits followed by Vb (e.g., 01Vb, 02Vb)
    match = re.search(r'0(\d+)vb', name_lower)
    if match:
        value = int(match.group(1))
        return value / 10.0
    
    return None


def _get_common_folder_name(viewer):
    """
    Extract a common folder name from the viewer's data:
    - If sweep_datasets exist, find the common prefix among all sample names
    - Otherwise, use the current sample_name
    
    Returns:
        str: Folder name to use for outputs
    """
    if hasattr(viewer, 'sweep_datasets') and viewer.sweep_datasets:
        # Extract all sample names
        names = [ds.get('sample_name', '') for ds in viewer.sweep_datasets]
        # Filter out empty names
        names = [n for n in names if n]
        if not names:
            return viewer.sample_name
        
        # Find common prefix
        if len(names) == 1:
            return names[0]
        
        # Find the longest common prefix
        common = os.path.commonprefix(names)
        # Remove trailing underscores or hyphens
        common = common.rstrip('_-')
        # If common prefix is too short or empty, use first name
        if len(common) < 3:
            return names[0]
        return common
    else:
        # Single dataset, use its sample name
        return viewer.sample_name


def ask_junction_width():
    """
    Show dialog to ask for junction width.
    
    Returns:
        float: Junction half-width in micrometers, or None if cancelled
    """
    root = tk.Tk()
    root.withdraw()
    root.update()
    try:
        width_um = simpledialog.askfloat(
            "Junction Width",
            "Half-width of junction (µm):",
            minvalue=1e-6,
            parent=root
        )
    except Exception:
        width_um = None
    finally:
        root.destroy()
    return width_um
    

def _maybe_close(fig):
    """
    Close a matplotlib figure only when running in a non-interactive / headless
    backend (e.g. 'Agg') or when matplotlib is not in interactive mode.
    """
    try:
        backend = mpl.get_backend().lower()
        # Only auto-close when backend is clearly headless (Agg).
        if backend.startswith('agg'):
            plt.close(fig)
    except Exception:
        try:
            plt.close(fig)
        except Exception:
            pass


def draw_scalebar(ax, pixel_size, length_um=1.0, height_px=5, color='white', font_size=12):
    """
    Draw a scale bar on the image axes.
    
    Args:
        ax: matplotlib axes object
        pixel_size: pixel size in meters
        length_um: length of scale bar in micrometers
        height_px: height of scale bar in pixels
        color: color of scale bar and text
        font_size: font size for scale bar label
    """
    pixel_size_um = pixel_size * 1e6
    length_px = length_um / pixel_size_um
    
    # Position in bottom-left corner
    x0 = 0.05 * ax.get_xlim()[1]
    y0 = 0.95 * ax.get_ylim()[0]
    
    # Draw rectangle
    rect = plt.Rectangle((x0, y0), length_px, height_px, color=color, clip_on=False)
    ax.add_patch(rect)
    
    # Add label
    ax.text(x0 + length_px / 2, y0 - height_px * 1.5, f"{length_um} µm",
            color=color, ha='center', va='bottom', fontsize=font_size, clip_on=False)

def extract_line_profile_data(line_coords, data_array, pixel_size):
    """
    Extract profile data along a line.
    
    Args:
        line_coords: ((x0, y0), (x1, y1)) tuple of endpoints
        data_array: 2D numpy array
        pixel_size: pixel size in meters
    
    Returns:
        tuple: (distances_um, profile_values) or (None, None) if invalid
    """
    if line_coords is None:
        return None, None
    
    (x0, y0), (x1, y1) = line_coords
    length = int(np.hypot(x1 - x0, y1 - y0))
    
    if length < 2:
        return None, None
    
    # Generate points along line
    x = np.linspace(x0, x1, length)
    y = np.linspace(y0, y1, length)
    
    # Clip to valid array bounds
    x = np.clip(x, 0, data_array.shape[1] - 1)
    y = np.clip(y, 0, data_array.shape[0] - 1)
    
    # Interpolate values
    profile_values = map_coordinates(data_array, [y, x], order=1, mode='nearest')
    
    # Calculate distances in micrometers
    distances_um = np.linspace(0, (length - 1) * pixel_size * 1e6, length)
    
    return distances_um, profile_values


def ask_junction_weight():
    """
    Show dialog to ask for EBIC weight used in junction detection.

    Returns:
        float: EBIC weight value or None if cancelled
    """
    root = tk.Tk()
    root.withdraw()
    root.update()
    try:
        weight = simpledialog.askfloat(
            "EBIC Weight",
            "Weight for EBIC influence in junction detection:",
            minvalue=0.1,
            initialvalue=10.0,
            parent=root
        )
    except Exception:
        weight = None
    finally:
        root.destroy()
    return weight


def safe_remove_colorbar(viewer):
    """
    Safely remove the colorbar associated with a viewer (if any).

    This handles both the Colorbar instance (`viewer.cbar`) and the
    dedicated axes (`viewer.cbar_ax`) that may have been created.
    """
    try:
        if getattr(viewer, 'cbar', None) is not None:
            try:
                viewer.cbar.remove()
            except Exception:
                pass
            viewer.cbar = None
    except Exception:
        pass

    try:
        if getattr(viewer, 'cbar_ax', None) is not None:
            try:
                viewer.cbar_ax.remove()
            except Exception:
                pass
            viewer.cbar_ax = None
    except Exception:
        pass


def plot_line_profile(viewer):
    """
    Plot line profile for current viewer state in a new window.
    Shows SEM data from frame 1 and current data from frame 2.
    
    Args:
        viewer: SEMViewer instance
    """
    if viewer.line_coords is None:
        print("No line coordinates available.")
        return
    
    # Extract SEM data from frame 1
    distances_um, sem_values = extract_line_profile_data(
        viewer.line_coords, 
        viewer.pixel_maps[0], 
        viewer.pixel_size
    )
    
    if distances_um is None:
        print("Failed to extract line profile data.")
        return
    
    # Extract current data from frame 2 if available
    current_values = None
    if len(viewer.current_maps) > 1 and viewer.current_maps[1] is not None:
        _, current_values = extract_line_profile_data(
            viewer.line_coords,
            viewer.current_maps[1],
            viewer.pixel_size
        )
    
    # Create plot
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    # Plot SEM data
    ax1.plot(distances_um, sem_values, color='tab:blue', label='SEM Contrast')
    ax1.set_xlabel("Distance (µm)")
    ax1.set_ylabel("Pixel Value (SEM)", color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Plot current data if available
    if current_values is not None:
        ax2 = ax1.twinx()
        ax2.plot(distances_um, current_values, color='tab:red', label='Current')
        ax2.set_ylabel("Current (nA)", color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        
        # Combine legends
        lines = ax1.get_lines() + ax2.get_lines()
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc='best')
    else:
        ax1.legend(loc='best')
        print("Warning: No current data available for frame 2.")
    
    plt.title("Line Profile - SEM (frame 1) + Current (frame 2)")
    plt.tight_layout()
    plt.show()


def save_line_profile_plot(viewer, save_root):
    """
    Save line profile plot to file.
    
    Args:
        viewer: SEMViewer instance
        save_root: root directory for saving
    """
    if viewer.line_coords is None:
        print("No line coordinates available.")
        return
    
    # Extract data
    distances_um, current_values = extract_line_profile_data(
        viewer.line_coords,
        viewer.current_maps[viewer.index],
        viewer.pixel_size
    )
    
    _, pixel_values = extract_line_profile_data(
        viewer.line_coords,
        viewer.pixel_maps[0],
        viewer.pixel_size
    )
    
    if distances_um is None or current_values is None or pixel_values is None:
        print("Failed to extract profile data for saving.")
        return
    
    # Create plot
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    # Plot pixel data
    ax1.plot(distances_um, pixel_values, color='tab:blue', label='Pixel (Frame 1)')
    ax1.set_xlabel("Distance (µm)")
    ax1.set_ylabel("Pixel Value (1st Frame)", color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    
    # Plot current data
    ax2 = ax1.twinx()
    ax2.plot(distances_um, current_values, color='tab:red', label='Current')
    ax2.set_ylabel("Current (nA)", color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    
    # Title and layout
    sample_name = viewer.sample_name if isinstance(viewer.sample_name, str) else viewer.sample_name[viewer.index]
    fig.suptitle(f"Line Profile: {os.path.basename(sample_name)}", fontsize=12)
    fig.tight_layout()
    
    # Save
    profile_save_path = os.path.join(
        save_root, 
        f"{viewer.sample_name}_frame{viewer.index + 1}_line_profile.png"
    )
    fig.savefig(profile_save_path, dpi=300, bbox_inches='tight')
    _maybe_close(fig)
    print(f"Saved line profile plot to {profile_save_path}")


# ==================== Analysis Functions ====================

def fit_perpendicular_profiles(viewer):
    """
    Fit diffusion profiles to perpendicular profiles.
    
    Args:
        viewer: SEMViewer instance with perpendicular_profiles attribute
    
    Returns:
        dict: Average diffusion lengths or None if no profiles available
    """
    if not hasattr(viewer, "perpendicular_profiles") or not viewer.perpendicular_profiles:
        print("No perpendicular profiles available. Draw them first.")
        return None
    
    pixel_size = getattr(viewer, 'pixel_size', 1e-6)
    extractor = DiffusionLengthExtractor(pixel_size, smoothing_sigma=1)
    extractor.load_profiles(viewer.perpendicular_profiles)
    extractor.fit_all_profiles()
    # extractor.visualize_fitted_profiles()
    # Also show log(EBIC) vs distance plots (saved to disk)
    # try:
    #     extractor.visualize_log_profiles()
    # except Exception as e:
    #     print("Error while plotting log profiles:", e)
    # extractor.visualize_depletion_regions()
    averages = extractor.compute_average_lengths(show_table=True)
    
    return averages


def fit_perpendicular_profiles_linear(viewer):
    """
    Fit two linear slopes on the log-scale of EBIC current for each perpendicular profile.

    This mirrors `fit_perpendicular_profiles` but instead of fitting exponentials
    it fits two straight lines to ln(current) on the left and right side of the
    intersection. Results are plotted and a summary list is returned.
    """
    if not hasattr(viewer, "perpendicular_profiles") or not viewer.perpendicular_profiles:
        print("No perpendicular profiles available. Draw them first.")
        return None

    # New workflow: use the DiffusionLengthExtractor linear fitting pipeline
    # which already implements linear-on-ln fitting, tail-cut truncation and
    # depletion-edge extraction. We will call the high-level method that
    # processes all profiles and then build a concise per-profile summary
    # containing left/right slopes (ln-domain), R² and the central non-linear
    # width (depletion_width).
    if not hasattr(viewer, "perpendicular_profiles") or not viewer.perpendicular_profiles:
        print("No perpendicular profiles available. Draw them first.")
        return None

    pixel_size = getattr(viewer, 'pixel_size', 1e-6)
    extractor = DiffusionLengthExtractor(pixel_size, smoothing_sigma=1)
    extractor.load_profiles(viewer.perpendicular_profiles)

    # Run the linear fitting pipeline for all profiles
    # Parameters for plateau detection and expansion:
    extractor.fit_all_profiles_linear(
        use_plateau_detection=True,   # Use derivative plateau method
        use_shifting=True,             # Enable iterative expansion
        gradient_window=7,             # Window size for derivative
        min_plateau_length=5,         # Minimum plateau size
        derivative_threshold=0.15,      # Max relative variation in derivative
        absolute_threshold=0.03,       # Max absolute variation (1/µm)
        max_expansion=1000,              # Max pixels to expand beyond plateau
        consecutive_drops=50,           # Stop after 30 consecutive R² drops
        junction_precision=False        # Start from junction to detect linearity onset precisely
    )

    summaries = []
    # Create output directory using common name from files
    folder_name = _get_common_folder_name(viewer)
    out_dir = os.path.join(os.getcwd(), 'depletion_plots', folder_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Saving linear fit results to: {out_dir}")

    for res in extractor.results:
        profile_idx = res.get('Profile')
        fit_sides = res.get('fit_sides', [])
        depletion = res.get('depletion', {})

        best_left = depletion.get('best_left_fit')
        best_right = depletion.get('best_right_fit')

        left_slope = best_left.get('slope') if best_left is not None else None
        left_r2 = best_left.get('r2') if best_left is not None else None
        right_slope = best_right.get('slope') if best_right is not None else None
        right_r2 = best_right.get('r2') if best_right is not None else None

        left_start = depletion.get('left_start')
        right_start = depletion.get('right_start')
        depletion_width = depletion.get('depletion_width')

        # Extract source_name from the profile if available (for sweep analysis)
        profile_source_name = None
        if profile_idx > 0 and profile_idx <= len(viewer.perpendicular_profiles):
            profile_entry = viewer.perpendicular_profiles[profile_idx - 1]
            profile_source_name = profile_entry.get('source_name', None)
        
        # Fallback to viewer.sample_name if no source_name in profile
        if not profile_source_name:
            profile_source_name = viewer.sample_name
        
        summaries.append({
            'profile': profile_idx,
            'sample_name': profile_source_name,
            'left_slope': left_slope,
            'left_r2': left_r2,
            'right_slope': right_slope,
            'right_r2': right_r2,
            'left_start': left_start,
            'right_start': right_start,
            'depletion_width': depletion_width
        })

        # Plot ln(current) with best linear fits and depletion markers
        try:
            profile_entry = extractor.profiles[profile_idx - 1]
            x = np.array(profile_entry.get('dist_um', []), dtype=float)
            y = np.array(profile_entry.get('current', []), dtype=float)
            base_idx = int(np.argmax(y))
            # Center x axis at the detected junction if available, otherwise at the peak
            intersection_idx = profile_entry.get('intersection_idx', None)
            if intersection_idx is not None and 0 <= int(intersection_idx) < len(x):
                ref = float(x[int(intersection_idx)])
                base_idx = int(intersection_idx)
            else:
                ref = float(x[base_idx])
            # Build ln(current) from raw data and also compute filtered version
            pos = y[y > 0]
            floor = max(np.min(pos) * 0.1, 1e-12) if pos.size > 0 else 1e-12
            y_safe = np.maximum(y, floor)
            ln_y = np.log(y_safe)

            # Compute filtered signal (used internally for fits) and its ln
            try:
                y_filtered = extractor.apply_low_pass_filter(y, visualize=False)
                y_filtered_safe = np.maximum(y_filtered, floor)
                ln_y_filtered = np.log(y_filtered_safe)
            except Exception:
                y_filtered = None
                ln_y_filtered = None

            # Create a 2-row figure: top = ln(Current) and fits, bottom = derivatives
            fig, (ax, ax_der) = plt.subplots(2, 1, sharex=True, figsize=(8, 6), gridspec_kw={'height_ratios': [2, 1]})
            x_plot = x - ref
            # Plot raw ln(Current)
            ax.plot(x_plot, ln_y, 'k.-', label='ln(Current) (raw)')
            # Plot filtered ln(Current) if available (smooth line)
            if ln_y_filtered is not None:
                ax.plot(x_plot, ln_y_filtered, color='tab:orange', lw=1.8, alpha=0.9, label='ln(Current) (filtered)')

            # Overlay best-left/right fits (transform their fit_curves to plotting coords)
            if best_left is not None:
                # map global_x_vals if present
                if 'global_x_vals' in best_left:
                    x_fit = np.array(best_left['global_x_vals'], dtype=float) - ref
                    # Use ln_fit_curve if available (plateau method), otherwise compute from fit_curve
                    if 'ln_fit_curve' in best_left:
                        y_fit = np.array(best_left['ln_fit_curve'], dtype=float)
                    else:
                        y_fit = np.log(np.maximum(np.array(best_left.get('fit_curve', [])), floor))
                else:
                    x_fit = -np.array(best_left.get('x_vals', []), dtype=float) - ref
                    y_fit = np.array(best_left.get('slope', 0.0)) * np.array(best_left.get('x_vals', []), dtype=float) + best_left.get('intercept', 0.0)
                ax.plot(x_fit, y_fit, 'b-', lw=2.0, label=f"Left fit (s={left_slope:.3g}, R²={left_r2:.2f})")
                if left_start is not None:
                    ax.axvline(left_start - ref, color='b', linestyle='--')

            if best_right is not None:
                if 'global_x_vals' in best_right:
                    x_fit = np.array(best_right['global_x_vals'], dtype=float) - ref
                    # Use ln_fit_curve if available (plateau method), otherwise compute from fit_curve
                    if 'ln_fit_curve' in best_right:
                        y_fit = np.array(best_right['ln_fit_curve'], dtype=float)
                    else:
                        y_fit = np.log(np.maximum(np.array(best_right.get('fit_curve', [])), floor))
                else:
                    x_fit = np.array(best_right.get('x_vals', []), dtype=float) - ref
                    y_fit = np.array(best_right.get('slope', 0.0)) * np.array(best_right.get('x_vals', []), dtype=float) + best_right.get('intercept', 0.0)
                ax.plot(x_fit, y_fit, 'r-', lw=2.0, label=f"Right fit (s={right_slope:.3g}, R²={right_r2:.2f})")
                if right_start is not None:
                    ax.axvline(right_start - ref, color='r', linestyle='--')

            # Shade depletion region
            if left_start is not None and right_start is not None:
                ax.axvspan(left_start - ref, right_start - ref, color='green', alpha=0.12)
            
            # Highlight detected plateau regions (if available)
            if best_left is not None and 'plateau_indices' in best_left:
                idx_start, idx_end = best_left['plateau_indices']
                if idx_start < len(x) and idx_end <= len(x):
                    plat_x_start = x[idx_start] - ref
                    plat_x_end = x[idx_end - 1] - ref
                    ax.axvspan(plat_x_start, plat_x_end, color='blue', alpha=0.08, label='Left plateau region')
            
            if best_right is not None and 'plateau_indices' in best_right:
                idx_start, idx_end = best_right['plateau_indices']
                if idx_start < len(x) and idx_end <= len(x):
                    plat_x_start = x[idx_start] - ref
                    plat_x_end = x[idx_end - 1] - ref
                    ax.axvspan(plat_x_start, plat_x_end, color='red', alpha=0.08, label='Right plateau region')

            ax.set_xlabel('Distance (µm)')
            ax.set_ylabel('ln(Current)')
            ax.set_title(f"Profile {profile_idx} — linear regimes and depletion width = {depletion_width if depletion_width is not None else float('nan'):.3g} µm")
            ax.legend(fontsize='small')
            ax.grid(True, linestyle='--', alpha=0.4)

            # --- Derivative panel: d(lnI)/dx ---
            # Use windowed gradient for smoother derivatives. Compute the
            # derivative from the RAW ln(Current) to match plateau detection.
            # IMPORTANT: These parameters are extracted from the extractor to match plateau detection
            gradient_window = extractor.plateau_params.get('gradient_window', 9)
            try:
                x_vals_plot = np.array(x_plot, dtype=float)
                if x_vals_plot.size > 1:
                    # Use RAW ln(y) to match plateau detection (no filtering)
                    source_ln = ln_y
                    dln_meas = gradient_with_window(x_vals_plot, source_ln, window=gradient_window)
                else:
                    dln_meas = np.zeros_like(x_vals_plot)
            except Exception:
                dln_meas = np.zeros_like(x_plot)

            ax_der.plot(x_plot, dln_meas, color='tab:gray', linestyle=':', label='dlnI/dx (meas)')

            # Overlay fit-derived dlnI/dx when available (from best_left/best_right)
            try:
                # helper to map fit arrays to plotting coordinates
                def _get_fit_derivative_mapped(fit_dict, side):
                    if fit_dict is None:
                        return None, None
                    # attempt to read global_x_vals first
                    if 'global_x_vals' in fit_dict:
                        fx = np.array(fit_dict['global_x_vals'], dtype=float) - ref
                    elif 'global_x' in fit_dict:
                        fx = np.array(fit_dict['global_x'], dtype=float) - ref
                    else:
                        # left-side stored x_vals are distances from left edge; mirror for plotting
                        if side == 'Left':
                            fx = -np.array(fit_dict.get('x_vals', []), dtype=float) - ref
                        else:
                            fx = np.array(fit_dict.get('x_vals', []), dtype=float) - ref

                    # derivative arrays - use windowed gradient
                    if 'fit_dlnI_dx' in fit_dict:
                        fdln = np.array(fit_dict['fit_dlnI_dx'], dtype=float)
                    else:
                        # try compute from fit_curve using windowed gradient
                        fc = np.array(fit_dict.get('fit_curve', []), dtype=float)
                        if fx.size > 1 and fc.size == fx.size:
                            try:
                                fdln = gradient_with_window(fx + ref, np.log(np.maximum(fc, 1e-12)), window=gradient_window)
                            except Exception:
                                fdln = None
                        else:
                            fdln = None
                    return fx, fdln

                left_fx, left_fdln = _get_fit_derivative_mapped(best_left, 'Left')
                right_fx, right_fdln = _get_fit_derivative_mapped(best_right, 'Right')

                # Plot derivatives directly (NO INTERPOLATION)
                if left_fx is not None and left_fdln is not None:
                    # sort
                    si = np.argsort(left_fx)
                    lf = left_fx[si]
                    ld = left_fdln[si]
                    # Plot directly without interpolation
                    ax_der.plot(lf, ld, 'b-', lw=1.6, label='dlnI/dx (left fit)')
                if right_fx is not None and right_fdln is not None:
                    si = np.argsort(right_fx)
                    rf = right_fx[si]
                    rd = right_fdln[si]
                    # Plot directly without interpolation
                    ax_der.plot(rf, rd, 'r-', lw=1.6, label='dlnI/dx (right fit)')
            except Exception:
                pass

            ax_der.set_xlabel('Distance (µm)')
            ax_der.set_ylabel('dlnI/dx (1/µm)')
            ax_der.grid(True, linestyle='--', alpha=0.4)
            ax_der.legend(fontsize='small')
            fig.tight_layout()
            # Save figure
            try:
                base = profile_entry.get('source_name', None) or f'profile_{profile_idx:02d}'
                import re
                base_safe = re.sub(r'[^A-Za-z0-9._-]', '_', str(base))
                out_path = os.path.join(out_dir, f'{base_safe}_profile_{profile_idx:02d}_linear.png')
                fig.savefig(out_path, dpi=200, bbox_inches='tight')
                print(f"Saved linear fit plot to {out_path}")
            except Exception:
                pass
            try:
                plt.show()
            except Exception:
                pass
            _maybe_close(fig)
        except Exception as e:
            # If any error occurs while preparing or plotting this profile, report and continue
            print(f"Failed plotting profile {profile_idx}: {e}")
            continue

    # Save summary CSV for all profiles with physical properties
    # Append mode: accumulate results across multiple analyses (especially for sweeps)
    import csv
    summary_path = os.path.join(out_dir, 'physical_properties_linear_fits.csv')
    try:
        # Check if file exists to determine if we need to write header
        file_exists = os.path.isfile(summary_path)
        
        with open(summary_path, 'a', newline='') as cf:
            writer = csv.writer(cf)
            
            # Write header only if file is new
            if not file_exists:
                writer.writerow(['sample_name', 'bias_V', 'profile', 'left_slope_1_per_um', 'left_r2', 'right_slope_1_per_um', 'right_r2', 
                               'left_edge_um', 'right_edge_um', 'depletion_width_um'])
            
            # Write data rows - each summary now contains its own sample_name (from profile source_name)
            for s in summaries:
                sample_name = s.get('sample_name', viewer.sample_name)
                bias_voltage = _extract_bias_voltage(sample_name)
                writer.writerow([sample_name, bias_voltage, s['profile'], s['left_slope'], s['left_r2'], s['right_slope'], s['right_r2'], 
                               s['left_start'], s['right_start'], s['depletion_width']])
        
        action = "Appended" if file_exists else "Saved"
        print(f"{action} physical properties (depletion range & slopes) to {summary_path}")
    except Exception as e:
        print(f"Failed to save physical properties CSV: {e}")

    # Also print a human-readable table to the console for quick inspection
    try:
        _print_linear_summary_table(summaries)
    except Exception as e:
        print(f"Failed to print linear summary table: {e}")

    return summaries


def _print_linear_summary_table(summaries):
    """Print an aligned table summary of linear fits to the console.

    Columns: profile, left_slope, left_r2, right_slope, right_r2, left_start, right_start, depletion_width
    """
    if not summaries:
        print("No summary data to print.")
        return

    # Define headers and column widths
    headers = ["profile", "left_slope", "left_r2", "right_slope", "right_r2", "left_start(um)", "right_start(um)", "depletion_width(um)"]
    col_w = [8, 12, 9, 12, 9, 15, 15, 18]

    # Print header
    hdr = "".join(h.center(w) for h, w in zip(headers, col_w))
    sep = "".join('-' * w for w in col_w)
    print('\nLinear-fit summary:')
    print(sep)
    print(hdr)
    print(sep)

    def fmt(x, prec=3):
        if x is None:
            return "-".rjust(12)
        try:
            return f"{float(x):.{prec}g}".rjust(12)
        except Exception:
            return str(x).rjust(12)

    for s in summaries:
        p = str(s.get('profile', '-'))
        ls = fmt(s.get('left_slope'))
        lr = fmt(s.get('left_r2'), prec=2)
        rs = fmt(s.get('right_slope'))
        rr = fmt(s.get('right_r2'), prec=2)
        lstart = fmt(s.get('left_start'))
        rstart = fmt(s.get('right_start'))
        dw = fmt(s.get('depletion_width'))

        row = p.center(col_w[0]) + ls.rjust(col_w[1]) + lr.rjust(col_w[2]) + rs.rjust(col_w[3]) + rr.rjust(col_w[4]) + lstart.rjust(col_w[5]) + rstart.rjust(col_w[6]) + dw.rjust(col_w[7])
        print(row)

    print(sep)
    print()


def detect_junction(viewer):
    """
    Detect junction along the drawn line and display it.
    
    Args:
        viewer: SEMViewer instance
    """
    if viewer.line_coords is None or viewer.manual_line_dense is None:
        print("Draw main line first")
        return
    
    # Ask for junction width
    width_um = ask_junction_width()
    if width_um is None:
        return
    
    # Calculate half-width in pixels
    half_width_px = max(1, int(round(width_um * 1e-6 / viewer.pixel_size)))
    
    # Extract ROI
    roi, roi_coords = extract_line_rectangle(
        viewer.pixel_maps[viewer.index],
        viewer.manual_line_dense,
        half_width_px
    )
    
    # Run junction detection
    analyzer = JunctionAnalyzer(pixel_size_m=viewer.pixel_size)
    results = analyzer.detect(roi, viewer.manual_line_dense)
    
    if results:
        best_result = results[0]
        detected_coords = best_result[1]
        
        # Remove old detected line if exists
        if hasattr(viewer, 'detected_line_obj') and viewer.detected_line_obj is not None:
            if viewer.detected_line_obj in viewer.ax.lines:
                viewer.detected_line_obj.remove()
        
        # Store and display detected junction
        viewer.detected_junction_line = detected_coords
        viewer.detected_line_obj = Line2D(
            detected_coords[:, 0],
            detected_coords[:, 1],
            color='green',
            linewidth=2
        )
        viewer.ax.add_line(viewer.detected_line_obj)
        viewer.fig.canvas.draw_idle()
        
        print("Junction detected and displayed in green.")
    else:
        print("Junction detection failed.")


def generate_perpendicular_profiles(viewer, num_lines, length_um):
    """
    Generate perpendicular profiles along the main line.
    
    Args:
        viewer: SEMViewer instance
        num_lines: number of perpendicular lines to generate
        length_um: length of each perpendicular line in micrometers
    """
    if viewer.line_coords is None:
        print("Draw main line first")
        return
    
    profiles = calculate_perpendicular_profiles(
        viewer.line_coords, 
        num_lines, 
        length_um,
        viewer.pixel_maps[0], 
        viewer.current_maps[1],
        detected_junction=getattr(viewer, 'detected_junction_line', None),
        source_name=getattr(viewer, 'sample_name', None)
    )
    
    viewer.perpendicular_profiles = profiles
    plot_perpendicular_profiles(profiles, ax=viewer.ax, fig=viewer.fig, source_name=getattr(viewer, 'sample_name', None), debug=False)
    # Also save log(EBIC) vs distance plots for the generated perpendicular profiles
    try:
        pixel_size = getattr(viewer, 'pixel_size', 1e-6)
        extractor = DiffusionLengthExtractor(pixel_size, smoothing_sigma=1)
        extractor.load_profiles(profiles)
        extractor.visualize_log_profiles()
    except Exception as e:
        print("Failed to create log plots for perpendicular profiles:", e)


def enable_windows_dpi_awareness():
    """Fix DPI scaling on Windows."""
    if sys.platform == "win32":
        try:
            from ctypes import windll
            windll.shcore.SetProcessDpiAwareness(1)
        except Exception as e:
            print("DPI awareness fix not applied:", e)
