# perpendicular.py
import numpy as np
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt

def calculate_perpendicular_profiles(line_coords, num_lines, length_um,
                                     sem_data, current_data,
                                     pixel_size_m=1e-6, detected_junction=None):
    """
    Calculate perpendicular profiles along a main line.
    Returns a list of dicts with SEM, current, and intersection info.
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

    return profiles


def plot_perpendicular_profiles(profiles, ax=None, fig=None):
    """
    Plot perpendicular lines and scrollable profile plots.
    """
    if not profiles: return
    if ax is not None and fig is not None:
        for prof in profiles:
            p_start, p_end = prof['line_coords']
            ax.plot([p_start[0], p_end[0]], [p_start[1], p_end[1]], 'r--', linewidth=1)
            if prof.get('intersection') is not None:
                inter = prof['intersection']
                ax.scatter(inter[0], inter[1], color='lime', s=50, marker='x', zorder=20)
        fig.canvas.draw_idle()

    # Plot each perpendicular profile
    import tkinter as tk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    win = tk.Toplevel()
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
        fig_prof, ax1 = plt.subplots(figsize=(6,3))
        ax1.plot(dist_um, sem_vals_norm, color='tab:blue', linewidth=2, label='SEM (norm)')
        ax2 = ax1.twinx()
        ax2.plot(dist_um, cur_vals, color='tab:red', linewidth=1.5, label='Current')
        if prof.get("intersection_idx") is not None:
            idx = prof["intersection_idx"]
            ax1.scatter(dist_um[idx], sem_vals_norm[idx], color='lime', s=50, marker='x', zorder=10)
            ax2.scatter(dist_um[idx], cur_vals[idx], color='lime', s=50, marker='x', zorder=10)
        ax1.set_xlabel("Distance (µm)")
        ax1.set_ylabel("SEM Contrast (norm)", color='tab:blue')
        ax2.set_ylabel("Current (nA)", color='tab:red')
        ax1.set_title(f"Perpendicular {prof['id']+1}")
        ax1.legend(loc='upper left')
        fig_prof.tight_layout()
        canvas_plot = FigureCanvasTkAgg(fig_prof, master=scroll_frame)
        widget = canvas_plot.get_tk_widget()
        widget.pack(pady=10)
