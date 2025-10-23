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
from matplotlib.lines import Line2D
from scipy.ndimage import map_coordinates

from .DiffLenExt import DiffusionLengthExtractor
from .Junction_Analyser import JunctionAnalyzer
from .ROI_extractor import extract_line_rectangle
from .perpendicular import calculate_perpendicular_profiles, plot_perpendicular_profiles


# ==================== Dialog Functions ====================

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


def ask_junction_weight():
    """
    Show dialog to ask for EBIC weight to combine with SEM gradient.

    Returns:
        float: weight multiplier for EBIC gradients, or None if cancelled
    """
    root = tk.Tk()
    root.withdraw()
    root.update()
    try:
        weight = simpledialog.askfloat(
            "EBIC Weight",
            "Weight applied to EBIC/current gradient (0 = ignore EBIC, 10 advised):",
            minvalue=0.0,
            parent=root
        )
    except Exception:
        weight = None
    finally:
        root.destroy()
    return weight


# ==================== Colorbar and Display Utilities ====================

def safe_remove_colorbar(viewer):
    """
    Safely remove colorbar from viewer if it exists.
    
    Args:
        viewer: SEMViewer instance
    """
    if getattr(viewer, 'cbar', None):
        try:
            viewer.cbar.remove()
        except Exception:
            pass
        viewer.cbar = None
    
    if getattr(viewer, 'cbar_ax', None):
        try:
            viewer.cbar_ax.remove()
        except Exception:
            pass
        viewer.cbar_ax = None


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


# ==================== Line Profile Extraction ====================

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


# ==================== Line Profile Plotting ====================

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
    plt.close(fig)
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
    
    extractor = DiffusionLengthExtractor(viewer.pixel_size, smoothing_sigma=1)
    extractor.load_profiles(viewer.perpendicular_profiles)
    extractor.fit_all_profiles()
    extractor.visualize_fitted_profiles()
    extractor.visualize_depletion_regions()
    averages = extractor.compute_average_lengths(show_table=True)
    
    return averages


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
        detected_junction=getattr(viewer, 'detected_junction_line', None)
    )
    
    viewer.perpendicular_profiles = profiles
    plot_perpendicular_profiles(profiles, ax=viewer.ax, fig=viewer.fig)


def enable_windows_dpi_awareness():
    """Fix DPI scaling on Windows."""
    if sys.platform == "win32":
        try:
            from ctypes import windll
            windll.shcore.SetProcessDpiAwareness(1)
        except Exception as e:
            print("DPI awareness fix not applied:", e)
