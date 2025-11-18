# sem_viewer.py
import os
import tkinter as tk
from tkinter import simpledialog
import numpy as np
import matplotlib

# Try to verify that Tk is usable (can create a root). If not, switch to Agg backend
# to avoid raising TclError in headless environments (CI/tests).
_has_tk = False
try:
    # importing tkinter succeeded above; try creating / destroying a root
    try:
        _tmp_root = tk.Tk()
        _tmp_root.destroy()
        _has_tk = True
    except Exception:
        _has_tk = False
except Exception:
    _has_tk = False

if not _has_tk:
    try:
        matplotlib.use('Agg')
    except Exception:
        pass

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.widgets import Button, Slider, TextBox
import matplotlib.patches as patches
from scipy.ndimage import map_coordinates
from scipy.stats import pearsonr
from .Junction_Analyser import JunctionAnalyzer
from .perpendicular import calculate_perpendicular_profiles, plot_perpendicular_profiles
from .helper_gui import fit_perpendicular_profiles, fit_perpendicular_profiles_linear
from .helper_gui import ask_junction_width, ask_junction_weight, draw_scalebar, safe_remove_colorbar
from .ROI_extractor import extract_line_rectangle

# Helper: close figures only in non-interactive/headless contexts
def _maybe_close(fig):
    try:
        backend = matplotlib.get_backend().lower()
        # Only auto-close when backend is clearly headless (Agg).
        if backend.startswith('agg'):
            plt.close(fig)
    except Exception:
        try:
            plt.close(fig)
        except Exception:
            pass

# ==================== SEM VIEWER ====================
class SEMViewer:
    def __init__(self, pixel_maps, current_maps, pixel_size, sample_name, frame_sizes=None, dpi=100, sweep_datasets=None, sweep_start_index=0):
        self.perpendicular_profiles = None
        self.pixel_maps = pixel_maps
        self.current_maps = current_maps
        self.pixel_size = pixel_size
        self.sample_name = sample_name
        # Backwards-compatibility/fallback: if processor didn't produce a separate
        # current_map for frame 2 but the TIFF includes a raw second frame, use it
        # as the current map so overlay and profile computations have data.
        try:
            if (self.current_maps is None or len(self.current_maps) <= 1 or self.current_maps[1] is None) and \
               (self.pixel_maps is not None and len(self.pixel_maps) > 1 and self.pixel_maps[1] is not None):
                # ensure list length
                if self.current_maps is None:
                    self.current_maps = [None] * (len(self.pixel_maps))
                while len(self.current_maps) < 2:
                    self.current_maps.append(None)
                self.current_maps[1] = self.pixel_maps[1]
                print(f"SEMViewer: no processed current_map for frame_2 — using raw frame_2 as fallback for '{self.sample_name}'")
        except Exception:
            # keep robust if inputs are unexpected
            pass
        self.index = 0
        self.map_type = 'current'
        self.overlay_mode = False
        self.line = None
        self.line_coords = None
        self._clicks = []
        self.frame_sizes = frame_sizes or [(pm.shape[1], pm.shape[0]) for pm in pixel_maps]
        self.junction_line = None  # Line2D object for manual line
        self.junction_coords = None  # Start/end coords of manual line
        self.junction_position = None  # Detected junction (x, y) along line
        # EBIC weight used by JunctionAnalyzer (default 10.0)
        self.ebic_weight = 10.0
        # Internal state to allow re-running detection when weight changes
        self._last_roi = None
        self._last_roi_current = None
        self._last_manual_line_dense = None
        self._last_half_width_px = None
        # flag used to avoid recursion when syncing slider/textbox
        self._weight_update_in_progress = False

        self.detected_junction_line = None
        self.detected_on_dataset_index = None

        # DPI and figure size to match the first frame
        width_px, height_px = self.frame_sizes[0]
        self.dpi = dpi
        fig_width_in = width_px / dpi
        fig_height_in = height_px / dpi
        self.fig = plt.figure(figsize=(fig_width_in, fig_height_in), dpi=dpi)
        # Axes positioned exactly for image, bottom-left fixed
        self.ax = self.fig.add_axes([0, 0, 1, 1])  # [left, bottom, width, height] in figure fraction
        self.cbar = None
        self.cbar_ax = None

        self.zoom_stack = []
        self.cid_scroll = self.fig.canvas.mpl_connect('scroll_event', self._on_scroll)
        self.zoom_rectangle = None
        self.cid_press = self.fig.canvas.mpl_connect('button_press_event', self._on_press)
        self.cid_release = self.fig.canvas.mpl_connect('button_release_event', self._on_release)
        self._dragging = False
        self.cid_motion = self.fig.canvas.mpl_connect('motion_notify_event', self._on_motion)

        self.current_colormaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'rainbow']
        self.current_cmap_index = 0
        self.zoom_mode_enabled = False
        self.zoom_rect_start = None
        self.zoom_rect = None

        # Initialize UI
        self._init_ui()
        self._update_display()
        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.junction_analyzer = JunctionAnalyzer(pixel_size_m=self.pixel_size)

        # Sweep datasets support: optional list of dicts with keys
        # 'pixel_maps','current_maps','pixel_size','sample_name','frame_sizes'
        self.sweep_datasets = sweep_datasets
        # Index within sweep_datasets currently displayed
        self.sweep_index = int(sweep_start_index) if sweep_datasets is not None else None
        # shifts (dx,dy) to map coordinates from reference (sweep_index) to each dataset
        self.sweep_shifts = None
        # per-dataset detection and perpendicular results (populated by Detect Sweep)
        self.sweep_detected_coords = None
        self.sweep_perpendicular_profiles = None
        # Debug mode: when True, automatically open the debug sweep view after actions
        self.debug_mode = False
        # If a sweep was provided, compute shifts relative to the initial shown dataset
        if self.sweep_datasets is not None:
            try:
                self._compute_sweep_shifts()
            except Exception:
                self.sweep_shifts = None

    def _init_ui(self):
        # Fixed image axes: occupy full height, leave space for buttons on the right
        img_width_frac = 0.85  # fraction of figure width for image
        img_height_frac = 1.0  # full height
        self.ax.set_position([0, 0, img_width_frac, img_height_frac])

        # Button layout parameters (two-column layout)
        spacing = 0.01
        button_h = 0.045
        side_width = max(0.10, 1.0 - img_width_frac - 3 * spacing)
        button_w = min(0.085, max(0.06, (side_width - spacing) / 2.0 - 0.005))
        left_button_offset = -0.01
        x_col1 = max(0.01, img_width_frac + spacing + left_button_offset)
        x_col2 = x_col1 + button_w + spacing
        button_y_offset = 0.03
        y_start = 0.88 + button_y_offset
        y1 = y_start
        y2 = y_start

        # Left column buttons
        ax_overlay = self.fig.add_axes([x_col1, y1, button_w, button_h])
        self.b_overlay = Button(ax_overlay, 'Overlay')
        self.b_overlay.on_clicked(self._toggle_overlay)
        y1 -= (button_h + spacing)

        ax_toggle = self.fig.add_axes([x_col1, y1, button_w, button_h])
        self.b_toggle = Button(ax_toggle, 'Toggle Map')
        self.b_toggle.on_clicked(self._toggle_map_type)
        y1 -= (button_h + spacing)

        ax_prev = self.fig.add_axes([x_col1, y1, button_w, button_h])
        self.b_prev = Button(ax_prev, 'Previous')
        self.b_prev.on_clicked(self._prev_frame)
        y1 -= (button_h + spacing)

        ax_next = self.fig.add_axes([x_col1, y1, button_w, button_h])
        self.b_next = Button(ax_next, 'Next')
        self.b_next.on_clicked(self._next_frame)
        y1 -= (button_h + spacing)

        ax_palette = self.fig.add_axes([x_col1, y1, button_w, button_h])
        self.b_palette = Button(ax_palette, 'Palette')
        self.b_palette.on_clicked(self._cycle_colormap)
        y1 -= (button_h + spacing)

        ax_perp = self.fig.add_axes([x_col1, y1, button_w, button_h])
        self.b_perp = Button(ax_perp, 'Perpendiculars')
        self.b_perp.on_clicked(self._show_perpendicular_input)
        y1 -= (button_h + spacing)

        ax_reset = self.fig.add_axes([x_col1, y1, button_w, button_h])
        self.b_reset = Button(ax_reset, 'Reset Zoom')
        self.b_reset.on_clicked(self._reset_zoom)
        y1 -= (button_h + spacing)

        ax_reset_overlays = self.fig.add_axes([x_col1, y1, button_w, button_h])
        self.b_reset_overlays = Button(ax_reset_overlays, 'Reset Lines')
        self.b_reset_overlays.on_clicked(self._reset_overlays)
        y1 -= (button_h + spacing)

        # Right column buttons
        ax_zoom = self.fig.add_axes([x_col2, y2, button_w, button_h])
        self.zoom_btn = Button(ax_zoom, 'Zoom')
        self.zoom_btn.on_clicked(self.enable_zoom)
        y2 -= (button_h + spacing)

        ax_fit = self.fig.add_axes([x_col2, y2, button_w, button_h])
        self.b_fit = Button(ax_fit, 'Fit Profiles exp')
        self.b_fit.on_clicked(self._fit_profiles)
        y2 -= (button_h + spacing)

        ax_fit_lin = self.fig.add_axes([x_col2, y2, button_w, button_h])
        self.b_fit_lin = Button(ax_fit_lin, 'Fit Profiles lin')
        self.b_fit_lin.on_clicked(self._fit_profiles_linear)
        y2 -= (button_h + spacing)

        ax_junc = self.fig.add_axes([x_col2, y2, button_w, button_h])
        self.b_junction = Button(ax_junc, 'Junction Detect')
        self.b_junction.on_clicked(self._detect_junction_button)
        y2 -= (button_h + spacing)

        # EBIC weight slider placed under junction detect in right column
        slider_h = 0.03
        ax_weight = self.fig.add_axes([x_col2, y2, button_w, slider_h])
        self.weight_slider = Slider(ax_weight, 'EBIC weight', 0.1, 100.0, valinit=self.ebic_weight)
        self.weight_slider.on_changed(self._on_weight_slider_change)
        textbox_w = 0.04
        textbox_x = x_col2 + button_w + 0.005
        ax_weight_val = self.fig.add_axes([textbox_x, y2, textbox_w, slider_h])
        self.weight_textbox = TextBox(ax_weight_val, '', initial=str(self.ebic_weight))
        self.weight_textbox.on_submit(self._on_weight_text_submit)
        y2 -= (slider_h + spacing)

        ax_observe = self.fig.add_axes([x_col2, y2, button_w, button_h])
        self.b_observe = Button(ax_observe, 'Observe Junction')
        self.b_observe.on_clicked(self._observe_detected_junction_button)
        y2 -= (button_h + spacing)

        ax_detect_sweep = self.fig.add_axes([x_col2, y2, button_w, button_h])
        self.b_detect_sweep = Button(ax_detect_sweep, 'Detect Sweep')
        self.b_detect_sweep.on_clicked(self._apply_detection_to_sweep)
        y2 -= (button_h + spacing)

        ax_debug = self.fig.add_axes([x_col2, y2, button_w, button_h])
        self.b_debug = Button(ax_debug, 'Debug Sweep')
        self.b_debug.on_clicked(self._debug_show_sweep)
        y2 -= (button_h + spacing)

        # Connect close event
        self.fig.canvas.mpl_connect('close_event', self._on_close)

        # If viewer was created with sweep datasets, add a small button to choose dataset
        if getattr(self, 'sweep_datasets', None) is not None and len(self.sweep_datasets) > 1:
            top_h = 0.03
            ax_choose = self.fig.add_axes([x_col1, 0.95, button_w, top_h])
            self.b_choose = Button(ax_choose, 'Choose Dataset')
            self.b_choose.on_clicked(self._choose_dataset_button)
            # Add a Debug Mode toggle near the chooser in second column
            ax_dbg_mode = self.fig.add_axes([x_col2, 0.95, button_w, top_h])
            self.b_debug_mode = Button(ax_dbg_mode, 'Debug: OFF')
            self.b_debug_mode.on_clicked(self._toggle_debug_mode)


    def _on_weight_slider_change(self, val):
        """Callback when the EBIC weight slider changes."""
        try:
            self.ebic_weight = float(val)
            # Provide quick user feedback
            print(f"EBIC weight set to {self.ebic_weight}")

            # Update textbox if present to reflect the new numeric value
            try:
                if hasattr(self, 'weight_textbox') and self.weight_textbox is not None:
                    # Update displayed text without triggering the textbox callback
                    try:
                        # TextBox.text_disp is the Text artist showing the value
                        self.weight_textbox.text_disp.set_text(str(self.ebic_weight))
                        # redraw canvas so change is visible
                        if hasattr(self, 'fig') and self.fig is not None:
                            self.fig.canvas.draw_idle()
                    except Exception:
                        # fallback to set_val if text_disp not available
                        try:
                            self.weight_textbox.set_val(str(self.ebic_weight))
                        except Exception:
                            pass
            except Exception:
                pass

            # Note: Do NOT re-run detection on every slider move; detection is triggered
            # only when the user presses the 'Junction Detect' button.
        except Exception:
            # ignore invalid values
            pass

    def _compute_sweep_shifts(self):
        """Compute pixel shifts for all sweep datasets relative to the current sweep_index dataset.

        Uses FFT-based cross-correlation to estimate integer pixel translations.
        """
        datasets = self.sweep_datasets
        ref_idx = int(self.sweep_index)
        # allow using EBIC/current maps to help registration when available
        # alignment weight controls contribution of current map (0 = ignore current, >0 includes it)
        align_w = getattr(self, 'alignment_ebic_weight', None)
        if align_w is None:
            # default to moderate weight (0.5)
            align_w = 0.5

        # reference images (may include current map)
        ref_pm = datasets[ref_idx].get('pixel_maps', [])
        ref_cm = datasets[ref_idx].get('current_maps', [])
        if not ref_pm:
            raise RuntimeError('Reference dataset has no pixel map')
        ref_img = ref_pm[0]
        ref_cur = ref_cm[1] if (ref_cm and len(ref_cm) > 1) else None

        shifts = []
        for d in datasets:
            pm = d.get('pixel_maps', [])
            cm = d.get('current_maps', [])
            if not pm:
                shifts.append((0.0, 0.0))
                continue
            img = pm[0]
            cur = cm[1] if (cm and len(cm) > 1) else None
            try:
                dx, dy = self._estimate_translation(ref_img, img, ref_current=ref_cur, cur_current=cur, weight_current=align_w)
            except Exception:
                dx, dy = 0.0, 0.0
            shifts.append((float(dx), float(dy)))
        self.sweep_shifts = shifts

    def _estimate_translation(self, ref, img, ref_current=None, cur_current=None, weight_current=0.5):
        """Estimate (dx, dy) shift to map ref coords to img coords.

        If current/EBIC maps are available, they are optionally combined with the SEM image
        to improve robustness: composite = norm(sem) + weight_current * norm(current).

        Returns (dx, dy) in pixels (floats when subpixel estimation is used).
        """
        # Build composite images to register (use SEM and optionally EBIC/current)
        def make_composite(sem_img, cur_img, w):
            a = np.asarray(sem_img, dtype=float)
            a = (a - np.mean(a)) / (np.std(a) + 1e-12)
            if cur_img is None or w == 0:
                return a
            c = np.asarray(cur_img, dtype=float)
            c = (c - np.mean(c)) / (np.std(c) + 1e-12)
            return a + (w * c)

        A = make_composite(ref, ref_current, weight_current)
        B = make_composite(img, cur_current, weight_current)

        # Crop to common minimal shape if sizes differ
        if A.shape != B.shape:
            minr = min(A.shape[0], B.shape[0])
            minc = min(A.shape[1], B.shape[1])
            A = A[:minr, :minc]
            B = B[:minr, :minc]

        # Try OpenCV phaseCorrelate for subpixel accuracy if available
        try:
            import cv2
            # phaseCorrelate expects float32 and uses log-polar optionally; use simple windowing
            A32 = np.float32(A)
            B32 = np.float32(B)
            # Optionally apply Hanning window to reduce edge effects
            try:
                win = cv2.createHanningWindow(A32.shape[::-1], cv2.CV_32F)
                Aw = A32 * win
                Bw = B32 * win
            except Exception:
                Aw = A32
                Bw = B32
            # returns (dx,dy) where positive dx means shift in x axis
            shift, resp = cv2.phaseCorrelate(Aw, Bw)
            # cv2.phaseCorrelate returns (shift_x, shift_y)
            dx = float(shift[0])
            dy = float(shift[1])
            return (dx, dy)
        except Exception:
            # Fallback: integer-pixel FFT cross-correlation
            fa = np.fft.fft2(A - np.mean(A))
            fb = np.fft.fft2(B - np.mean(B))
            cross = np.fft.ifft2(fa * np.conj(fb))
            cross_abs = np.abs(cross)
            maxpos = np.unravel_index(np.argmax(cross_abs), cross_abs.shape)
            shift_y, shift_x = maxpos
            # convert to signed shifts
            if shift_x > cross.shape[1] // 2:
                shift_x -= cross.shape[1]
            if shift_y > cross.shape[0] // 2:
                shift_y -= cross.shape[0]
            return (float(shift_x), float(shift_y))

    def _choose_dataset_button(self, event=None):
        # Open small dialog to choose dataset
        try:
            win = tk.Toplevel()
            win.title("Choose dataset to display")
            listbox = tk.Listbox(win, selectmode=tk.SINGLE, width=60)
            listbox.pack(padx=10, pady=10)
            for i, d in enumerate(self.sweep_datasets):
                listbox.insert(tk.END, f"{i}: {d.get('sample_name', 'dataset'+str(i))}")

            def on_ok():
                sel = listbox.curselection()
                if not sel:
                    tk.messagebox.showwarning("No selection", "Select a dataset to display")
                    return
                idx = sel[0]
                win.destroy()
                self._switch_to_dataset(idx)

            tk.Button(win, text="Open", command=on_ok).pack(pady=6)
            win.transient(self.fig.canvas.get_tk_widget().winfo_toplevel())
            win.grab_set()
            win.wait_window()
        except Exception as e:
            print(f"Failed to open dataset chooser: {e}")

    def _switch_to_dataset(self, idx):
        """Switch the viewer to display dataset at sweep_datasets[idx]."""
        try:
            ds = self.sweep_datasets[idx]
            # update internal maps and meta
            self.pixel_maps = ds['pixel_maps']
            self.current_maps = ds['current_maps']
            self.pixel_size = ds['pixel_size']
            self.sample_name = ds['sample_name']
            self.frame_sizes = ds.get('frame_sizes', self.frame_sizes)
            self.sweep_index = int(idx)

            # refresh display
            self._update_display()
            # If debug mode is on, show the debug sweep view to inspect alignment
            try:
                if getattr(self, 'debug_mode', False):
                    self._debug_show_sweep()
            except Exception:
                pass
        except Exception as e:
            print(f"Failed to switch dataset: {e}")

    def _get_displayed_detected_coords(self):
        """Return detected junction coordinates mapped to coordinates of the currently displayed dataset."""
        # If per-dataset sweep detections exist, prefer those for the displayed dataset
        try:
            if getattr(self, 'sweep_datasets', None) is not None and getattr(self, 'sweep_detected_coords', None) is not None:
                target_idx = int(self.sweep_index) if self.sweep_index is not None else 0
                if 0 <= target_idx < len(self.sweep_detected_coords):
                    det = self.sweep_detected_coords[target_idx]
                    if det is not None:
                        return np.asarray(det)
        except Exception:
            pass

        if not hasattr(self, 'detected_junction_line') or self.detected_junction_line is None:
            return None
        coords = np.asarray(self.detected_junction_line)
        if self.sweep_datasets is None or self.sweep_shifts is None:
            return coords
        try:
            # mapping: coords_in_target = coords_in_detected + (shift[target] - shift[detected])
            det_idx = getattr(self, 'detected_on_dataset_index', self.sweep_index)
            target_idx = self.sweep_index
            sx_t, sy_t = self.sweep_shifts[target_idx]
            sx_d, sy_d = self.sweep_shifts[det_idx]
            dx = sx_t - sx_d
            dy = sy_t - sy_d
            mapped = coords + np.array([dx, dy])
            return mapped
        except Exception:
            return coords

    def _on_weight_text_submit(self, text):
        """Callback when user types a value into the EBIC weight textbox and presses Enter."""
        try:
            val = float(text)
        except Exception:
            # ignore invalid input
            return

        # Clamp to slider range if available
        try:
            if hasattr(self, 'weight_slider') and self.weight_slider is not None:
                val = max(self.weight_slider.valmin, min(self.weight_slider.valmax, val))
                # Setting slider value will call _on_weight_slider_change and update ebic_weight
                self.weight_slider.set_val(val)
            else:
                self.ebic_weight = val
        except Exception:
            self.ebic_weight = val

    def enable_zoom(self, event):
        """Toggle zoom mode on/off"""
        self.zoom_mode_enabled = not self.zoom_mode_enabled

        if self.zoom_mode_enabled:
            self.zoom_btn.label.set_text("Zooming...")
            print("Zoom mode enabled: Click and drag to draw a zoom rectangle")
        else:
            self.zoom_btn.label.set_text("Zoom")
            # Clean up any existing rectangle
            if self.zoom_rect:
                self.zoom_rect.remove()
                self.zoom_rect = None
            self.zoom_rect_start = None

        self.fig.canvas.draw_idle()

    def _get_current_data(self):
        return self.pixel_maps if self.map_type == 'pixel' else self.current_maps

    def _toggle_overlay(self, event):
        self.overlay_mode = not self.overlay_mode
        self._update_display()

    def _toggle_map_type(self, event):
        self.map_type = 'pixel' if self.map_type == 'current' else 'current'
        self._update_display()

    def _next_frame(self, event):
        self.index = (self.index + 1) % len(self.pixel_maps)
        self._update_display()

    def _prev_frame(self, event):
        self.index = (self.index - 1) % len(self.pixel_maps)
        self._update_display()

    def _update_display(self):
        # Store current view limits before clearing
        current_xlim = self.ax.get_xlim()
        current_ylim = self.ax.get_ylim()
        had_zoom = len(current_xlim) == 2 and current_xlim != (0.0, 1.0)

        # Clear previous state while preserving figure properties
        self.ax.clear()
        safe_remove_colorbar(self)

        # Reset ticks and labels
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        # Handle empty data case
        if not self.pixel_maps or not self.current_maps:
            self.ax.set_title("No data available")
            self.fig.canvas.draw_idle()
            return

        try:
            if self.overlay_mode:
                # Overlay mode implementation with diagnostics
                if len(self.pixel_maps) < 1 or len(self.current_maps) < 2:
                    print(f"Overlay requested but not enough frames: pixel_maps={len(self.pixel_maps)}, current_maps={len(self.current_maps)}")
                    self.ax.set_title("Overlay: Not enough frames.")
                else:
                    pixel = self.pixel_maps[self.index]
                    # Prefer the processed current map, but if it wasn't produced (None),
                    # fall back to the raw second frame (pixel_maps[1]) which may already
                    # contain EBIC values in some TIFF varieties.
                    current = None
                    if len(self.current_maps) > 1:
                        current = self.current_maps[1]
                    if current is None and len(self.pixel_maps) > 1:
                        # fallback: use raw second frame as current map
                        current = self.pixel_maps[1]
                        print(f"Overlay fallback: using raw frame_2 (pixel_maps[1]) as current map for '{self.sample_name}'")

                    if pixel is not None:
                        self.im = self.ax.imshow(
                            pixel, cmap='gray',
                            extent=[0, pixel.shape[1], pixel.shape[0], 0]
                        )
                        if not had_zoom:  # Only set full view if not zoomed
                            self.ax.set_xlim(0, pixel.shape[1])
                            self.ax.set_ylim(pixel.shape[0], 0)

                    # If current is present, overlay it; otherwise warn and continue showing SEM
                    if current is not None and pixel is not None:
                        # Create colorbar axis if it doesn't exist
                        if self.cbar_ax is None or not self.cbar_ax in self.fig.axes:
                            self.cbar_ax = self.fig.add_axes([0.82, 0.1, 0.02, 0.3])

                        self.im = self.ax.imshow(
                            current,
                            cmap=self.current_colormaps[self.current_cmap_index],
                            alpha=0.6,
                            extent=[0, pixel.shape[1], pixel.shape[0], 0]
                        )
                        self.cbar = self.fig.colorbar(self.im, cax=self.cbar_ax)
                        self.cbar.set_label("Current (nA)")
                    else:
                        print(f"Overlay: current map missing for '{self.sample_name}' (current_maps[1] is None). Showing pixel map only.")

                    self.ax.set_title(f"{self.sample_name} - Overlay: Pixel Map + Current Map")

            else:
                # Single map mode implementation
                data_list = self._get_current_data()
                data = data_list[self.index]

                if data is None:
                    if self.map_type == 'current':
                        self.map_type = 'pixel'
                        data = self.pixel_maps[self.index]
                        title = f"{self.sample_name} - (No current map) Showing pixel map - Frame {self.index + 1}"
                    else:
                        title = f"{self.sample_name} - No data available - Frame {self.index + 1}"
                    self.ax.set_title(title)

                if data is not None:
                    # === Frame 0 = pixel map → grayscale, no colorbar ===
                    if self.index == 0 or self.map_type == 'pixel':
                        self.im = self.ax.imshow(
                            data, cmap='gray',
                            extent=[0, data.shape[1], data.shape[0], 0]
                        )
                    else:
                        # Current maps (frames >= 1) → colorbar
                        cmap = self.current_colormaps[self.current_cmap_index]
                        if self.cbar_ax is None or not self.cbar_ax in self.fig.axes:
                            self.cbar_ax = self.fig.add_axes([0.82, 0.1, 0.02, 0.3])

                        self.im = self.ax.imshow(
                            data, cmap=cmap,
                            extent=[0, data.shape[1], data.shape[0], 0]
                        )
                        self.cbar = self.fig.colorbar(self.im, cax=self.cbar_ax)
                        self.cbar.set_label("Current (nA)")

                    # Set proper view limits based on image dimensions only if not zoomed
                    if not had_zoom:
                        self.ax.set_xlim(0, data.shape[1])
                        self.ax.set_ylim(data.shape[0], 0)  # Inverted y-axis

                    self.ax.set_title(
                        f"{self.sample_name} - "
                        f"{'Pixel' if self.index == 0 else self.map_type.capitalize()} Map - Frame {self.index + 1}"
                    )

            # Restore zoom state if we had one
            if had_zoom:
                self.ax.set_xlim(current_xlim)
                self.ax.set_ylim(current_ylim)

            # Restore line if it exists
            if self.line_coords:
                (x0, y0), (x1, y1) = self.line_coords
                self.line = Line2D([x0, x1], [y0, y1], color='red', linewidth=2)
                self.ax.add_line(self.line)

            # Draw scale bar
            draw_scalebar(self.ax,self.pixel_size)

            # Draw detected junction (mapped to currently displayed dataset) if available
            try:
                if hasattr(self, 'detected_junction_line') and self.detected_junction_line is not None:
                    # remove previous visual object if present
                    try:
                        if hasattr(self, 'detected_line_obj') and self.detected_line_obj is not None:
                            try:
                                self.detected_line_obj.remove()
                            except Exception:
                                pass
                    except Exception:
                        pass

                    mapped = self._get_displayed_detected_coords()
                    if mapped is None:
                        mapped = self.detected_junction_line

                    self.detected_line_obj = Line2D(mapped[:, 0], mapped[:, 1], color='green', linewidth=2)
                    try:
                        self.ax.add_line(self.detected_line_obj)
                    except Exception:
                        pass
            except Exception:
                pass

            self.fig.canvas.draw_idle()

        except Exception as e:
            print(f"Error in display update: {str(e)}")
            import traceback
            traceback.print_exc()

    def show(self):
        plt.show()

    def _cycle_colormap(self, event=None):
        if not self.current_colormaps:
            return
        self.current_cmap_index = (self.current_cmap_index + 1) % len(self.current_colormaps)
        self._update_display()

    def _on_click(self, event):
        if event.inaxes != self.ax or event.button != 1:
            return

        if self.zoom_mode_enabled:
            return

        if not hasattr(self, '_clicks'):
            self._clicks = []

        if len(self._clicks) == 0:
            # First click
            self._clicks.append((event.xdata, event.ydata))
            self._dragging = True

            if hasattr(self, 'click_marker') and self.click_marker in self.ax.collections:
                self.click_marker.remove()
            self.click_marker = self.ax.scatter(
                [event.xdata], [event.ydata],
                color='red', s=50, marker='o', zorder=10)

        elif len(self._clicks) == 1:
            # Second click - finish line
            self._dragging = False

            if hasattr(self, '_temp_line') and self._temp_line in self.ax.lines:
                self._temp_line.remove()

            x0, y0 = self._clicks[0]
            x1, y1 = event.xdata, event.ydata
            self.line_coords = ((x0, y0), (x1, y1))

            if self.line and self.line in self.ax.lines:
                self.line.remove()

            self.line = Line2D([x0, x1], [y0, y1],
                               color='red',
                               linewidth=2)
            self.ax.add_line(self.line)

            # === Generate dense manual line ===
            line_length = int(np.ceil(np.hypot(x1 - x0, y1 - y0)))
            xs = np.linspace(x0, x1, line_length)
            ys = np.linspace(y0, y1, line_length)
            self.manual_line_dense = np.column_stack([xs, ys])

            # Cleanup click markers
            self._clicks = []
            if hasattr(self, 'click_marker'):
                self.click_marker.remove()
                del self.click_marker

            self.fig.canvas.draw_idle()

            # Manual line created. Do not auto-run detection here.
            # Clear any previously stored ROI so detection will be re-extracted when
            # the user presses the 'Junction Detect' button.
            print("Manual line drawn. Press 'Junction Detect' to run detection (uses EBIC weight slider).")
            self._last_roi = None
            self._last_roi_current = None
            self._last_manual_line_dense = self.manual_line_dense
            self._last_half_width_px = None


    def _on_close(self, event):
        print("Figure closing, saving images and data...")

        # Prepare save folder
        save_root = os.path.join("tiff_test_output", self.sample_name, "saved_on_close")
        os.makedirs(save_root, exist_ok=True)

        # Save current displayed frame image with overlay line if any
        img_save_path = os.path.join(save_root, f"{self.sample_name}_frame{self.index + 1}_overlay.png")

        # Draw the overlay line on a copy of the current image and save
        fig, ax = plt.subplots()
        data = (self.pixel_maps if self.map_type == 'pixel' else self.current_maps)[self.index]

        if data is None:
            print("No data to save for the current frame.")
            _maybe_close(fig)
            return

        cmap = 'gray' if self.map_type == 'pixel' else self.current_colormaps[self.current_cmap_index]
        ax.imshow(data, cmap=cmap)
        if self.line is not None:
            ax.add_line(Line2D(self.line.get_xdata(), self.line.get_ydata(), color='red'))

        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(img_save_path)
        _maybe_close(fig)
        print(f"Saved overlay image to {img_save_path}")

    def _on_scroll(self, event):
        if event.inaxes != self.ax:
            return

        x_min, x_max = self.ax.get_xlim()
        y_min, y_max = self.ax.get_ylim()

        zoom_factor = 1.2 if event.button == 'up' else 0.8
        cx, cy = event.xdata, event.ydata

        new_width = (x_max - x_min) * zoom_factor
        new_height = (y_max - y_min) * zoom_factor

        new_xmin = cx - new_width / 2
        new_xmax = cx + new_width / 2
        new_ymin = cy - new_height / 2
        new_ymax = cy + new_height / 2

        # Save current view for reset
        self.zoom_stack.append((self.ax.get_xlim(), self.ax.get_ylim()))

        # Apply new limits
        self.ax.set_xlim(new_xmin, new_xmax)
        self.ax.set_ylim(new_ymin, new_ymax)

        # 🔑 Force equal aspect ratio (no stretching)
        self.ax.set_aspect('equal', adjustable='box')

        self.fig.canvas.draw_idle()

    def _on_press(self, event):
        if event.inaxes != self.ax:
            return

        width, height = self.frame_sizes[self.index]

        # Left-click zoom rectangle
        if self.zoom_mode_enabled and event.button == 1:
            x = max(0, min(event.xdata, width))
            y = max(0, min(event.ydata, height))
            self.zoom_rect_start = (x, y)
            return

        # Right-click drag
        if event.button == 3:
            x = max(0, min(event.xdata, width))
            y = max(0, min(event.ydata, height))
            self._drag_start = (x, y)

    def _on_release(self, event):
        if event.inaxes != self.ax:
            return

        width, height = self.frame_sizes[self.index]

        # Left-click zoom rectangle
        if self.zoom_mode_enabled and self.zoom_rect_start and event.button == 1:
            x0, y0 = self.zoom_rect_start
            x1 = max(0, min(event.xdata, width))
            y1 = max(0, min(event.ydata, height))

            if abs(x1 - x0) > 5 and abs(y1 - y0) > 5:
                self.zoom_stack.append((self.ax.get_xlim(), self.ax.get_ylim()))
                self.ax.set_xlim(min(x0, x1), max(x0, x1))
                self.ax.set_ylim(max(y0, y1), min(y0, y1))

            if self.zoom_rect:
                self.zoom_rect.remove()
                self.zoom_rect = None
            self.zoom_rect_start = None
            self.fig.canvas.draw_idle()
            return

        # Right-click drag zoom
        if hasattr(self, "_drag_start") and event.button == 3:
            x0, y0 = self._drag_start
            x1 = max(0, min(event.xdata, width))
            y1 = max(0, min(event.ydata, height))

            if abs(x1 - x0) > 2 and abs(y1 - y0) > 2:
                self.zoom_stack.append((self.ax.get_xlim(), self.ax.get_ylim()))
                self.ax.set_xlim(min(x0, x1), max(x0, x1))
                self.ax.set_ylim(max(y0, y1), min(y0, y1))
                self.fig.canvas.draw_idle()
            del self._drag_start

    def _on_motion(self, event):
        if event.inaxes != self.ax:
            return

        width, height = self.frame_sizes[self.index]

        # === Case 1: zoom rectangle ===
        if self.zoom_mode_enabled and self.zoom_rect_start:
            if self.zoom_rect:
                self.zoom_rect.remove()

            x0, y0 = self.zoom_rect_start
            x1 = max(0, min(event.xdata, width))
            y1 = max(0, min(event.ydata, height))

            self.zoom_rect = patches.Rectangle(
                (min(x0, x1), min(y0, y1)),
                abs(x1 - x0), abs(y1 - y0),
                linewidth=1, edgecolor='r', facecolor='none', linestyle='--'
            )
            self.ax.add_patch(self.zoom_rect)
            self.fig.canvas.draw_idle()
            return

        # === Case 2: live preview of line ===
        if self._dragging and len(self._clicks) == 1:
            x0, y0 = self._clicks[0]
            x1, y1 = event.xdata, event.ydata
            if x1 is None or y1 is None:
                return

            # Remove previous temp line
            if hasattr(self, '_temp_line') and self._temp_line in self.ax.lines:
                self._temp_line.remove()

            # Draw new temp line
            self._temp_line = Line2D([x0, x1], [y0, y1],
                                     color='red', linewidth=1, linestyle='--')
            self.ax.add_line(self._temp_line)
            self.fig.canvas.draw_idle()

    def _reset_zoom(self, event=None):
        self.zoom_stack.clear()

        data_list = self._get_current_data()
        if not data_list or data_list[self.index] is None:
            return

        data = data_list[self.index]

        # Reset to full image extents
        self.ax.set_xlim(0, data.shape[1])
        self.ax.set_ylim(data.shape[0], 0)  # inverted y-axis

        self._update_display()
        self.fig.canvas.draw_idle()

    def _show_perpendicular_input(self, event):
        """Ask user for number of perpendiculars and length, then calculate and plot them."""
        from tkinter import simpledialog, Tk

        # Use a temporary root for dialogs, then destroy it
        root = Tk()
        root.withdraw()  # hide the root window
        root.update()  # process events

        try:
            num_lines = simpledialog.askinteger(
                "Input", "Number of perpendicular lines:", parent=root, minvalue=1, maxvalue=100
            )
            length_um = simpledialog.askfloat(
                "Input", "Length of each perpendicular line (µm):", parent=root, minvalue=0.1, maxvalue=1e5
            )
        except Exception:
            print("Invalid input. Operation canceled.")
            root.destroy()
            return

        root.destroy()

        if num_lines is None or length_um is None:
            print("Operation canceled by user.")
            return

        if self.line_coords is None:
            print("Main line not drawn.")
            return

        # If we have a sweep, compute and save perpendiculars for every dataset
        if getattr(self, 'sweep_datasets', None) is not None and len(self.sweep_datasets) > 0:
            print(f"Computing perpendicular profiles for all {len(self.sweep_datasets)} sweep datasets...")
            # Determine base dense line representation
            base_line = None
            if isinstance(self.line_coords, np.ndarray):
                base_line = self.line_coords
            else:
                # create dense line from tuple
                (x0, y0), (x1, y1) = self.line_coords
                line_length = int(np.ceil(np.hypot(x1 - x0, y1 - y0)))
                xs = np.linspace(x0, x1, line_length)
                ys = np.linspace(y0, y1, line_length)
                base_line = np.column_stack([xs, ys])

            # For each dataset, map the base line using sweep_shifts (if available) and compute profiles
            for i, ds in enumerate(self.sweep_datasets):
                try:
                    # map line to dataset coords
                    mapped_line = base_line.copy()
                    if getattr(self, 'sweep_shifts', None) is not None and getattr(self, 'sweep_index', None) is not None:
                        # mapping from reference (sweep_index) to target i
                        ref_idx = int(self.sweep_index)
                        sx_t, sy_t = self.sweep_shifts[i]
                        sx_r, sy_r = self.sweep_shifts[ref_idx]
                        dx = sx_t - sx_r
                        dy = sy_t - sy_r
                        mapped_line = mapped_line + np.array([dx, dy])

                    sem_data = ds.get('pixel_maps', [None])[0]
                    cur_data = None
                    cm_list = ds.get('current_maps', [])
                    if cm_list and len(cm_list) > 1:
                        cur_data = cm_list[1]

                    # If we have per-dataset detected junction coords, use them as detected_junction
                    detected_junction = None
                    if getattr(self, 'sweep_detected_coords', None) is not None:
                        try:
                            detected_junction = self.sweep_detected_coords[i]
                        except Exception:
                            detected_junction = None
                    # fallback: map global detected_junction_line to this dataset
                    if detected_junction is None and getattr(self, 'detected_junction_line', None) is not None:
                        try:
                            src_idx = getattr(self, 'detected_on_dataset_index', self.sweep_index)
                            if self.sweep_shifts is not None and src_idx is not None:
                                sx_i, sy_i = self.sweep_shifts[i]
                                sx_s, sy_s = self.sweep_shifts[src_idx]
                                dx = sx_i - sx_s
                                dy = sy_i - sy_s
                                detected_junction = np.asarray(self.detected_junction_line) + np.array([dx, dy])
                            else:
                                detected_junction = np.asarray(self.detected_junction_line)
                        except Exception:
                            detected_junction = None

                    # Compute perpendicular profiles for this dataset
                    if sem_data is None:
                        print(f"Dataset {i} missing SEM data, skipping perpendiculars.")
                        continue

                    profiles = calculate_perpendicular_profiles(mapped_line, int(num_lines), float(length_um), sem_data, cur_data, pixel_size_m=ds.get('pixel_size', self.pixel_size), detected_junction=detected_junction, source_name=ds.get('sample_name'))

                    # Plot & save profiles using shared utility (it saves PNG/CSV)
                    try:
                        # If this dataset is the one currently displayed, draw overlays on the main viewer
                        cur_display_idx = int(self.sweep_index) if getattr(self, 'sweep_index', None) is not None else 0
                        if i == cur_display_idx and hasattr(self, 'ax') and hasattr(self, 'fig'):
                            plot_perpendicular_profiles(profiles, ax=self.ax, fig=self.fig, source_name=ds.get('sample_name'), debug=False)
                        else:
                            # just open/save the perpendicular plots for other datasets
                            plot_perpendicular_profiles(profiles, ax=None, fig=None, source_name=ds.get('sample_name'), debug=False)
                    except Exception as e:
                        print(f"Failed to plot perpendiculars for dataset {i}: {e}")

                except Exception as e:
                    print(f"Failed perpendiculars for dataset {i}: {e}")

            print("Saved perpendicular plots for all sweep datasets.")
            # Also update in-memory perpendicular_profiles for the current viewer if available
            try:
                if getattr(self, 'sweep_perpendicular_profiles', None) is not None:
                    # already stored by detection pass; nothing to change
                    pass
            except Exception:
                pass
            return

        # === Calculate perpendicular profiles for current dataset ===
        profiles = self._calculate_perpendicular_profiles(num_lines, length_um, line_coords=self.line_coords)

        # === Plot perpendicular profiles ===
        self._plot_perpendicular_profiles(profiles)

    def _calculate_perpendicular_profiles(self, num_lines, length_um, line_coords=None):
        """
        Calculate perpendicular profiles along the manual line.
        Adds the intersection with detected junction if available.
        """
        if line_coords is None:
            line_coords = self.line_coords  # fallback

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

        main_vec = np.array([x1 - x0, y1 - y0], dtype=float)
        main_len = np.linalg.norm(main_vec)
        if main_len == 0:
            print("Line is too short.")
            return []

        main_unit = main_vec / main_len
        perp_unit = np.array([-main_unit[1], main_unit[0]])
        length_px = length_um / (self.pixel_size * 1e6)  # µm -> px

        # Pick points along dense line
        distances = np.linspace(0, len(dense_line) - 1, num_lines).astype(int)

        # Use the currently displayed SEM frame (respect self.index)
        try:
            sem_data = self.pixel_maps[self.index]
        except Exception:
            sem_data = self.pixel_maps[0] if self.pixel_maps else None

        # Prefer processed current map (index 1) if present, otherwise fallback to pixel_maps[1]
        current_data = None
        try:
            if getattr(self, 'current_maps', None) and len(self.current_maps) > 1 and self.current_maps[1] is not None:
                current_data = self.current_maps[1]
            elif len(self.pixel_maps) > 1 and self.pixel_maps[1] is not None:
                current_data = self.pixel_maps[1]
        except Exception:
            current_data = None

        if sem_data is None:
            print("Missing SEM data.")
            return []

        profiles = []
        for i, idx in enumerate(distances):
            center = dense_line[idx]
            p_start = center - perp_unit * (length_px / 2)
            p_end = center + perp_unit * (length_px / 2)

            x = np.linspace(p_start[0], p_end[0], int(length_px))
            y = np.linspace(p_start[1], p_end[1], int(length_px))
            x = np.clip(x, 0, sem_data.shape[1] - 1)
            y = np.clip(y, 0, sem_data.shape[0] - 1)

            sem_values = map_coordinates(sem_data, [y, x], order=1, mode='nearest')
            current_values = map_coordinates(current_data, [y, x], order=1, mode='nearest')
            dist_um_line = np.linspace(-length_um / 2, length_um / 2, len(sem_values))

            # --- Find intersection with detected junction ---
            intersection = None
            intersection_idx = None
            if hasattr(self, 'detected_junction_line') and self.detected_junction_line is not None:
                intersection, intersection_idx = self._find_perpendicular_junction_intersection(
                    p_start, p_end, self.detected_junction_line, len(sem_values)
                )

            profiles.append({
                "id": i,
                "dist_um": dist_um_line,
                "current": current_values,
                "sem": sem_values,
                "line_coords": (p_start, p_end),
                "intersection": intersection,
                "intersection_idx": intersection_idx
            })

        self.perpendicular_profiles = profiles
        return profiles


    def _find_perpendicular_junction_intersection(self, p_start, p_end, junction_line, profile_length):
        line_vec = p_end - p_start
        line_len2 = np.sum(line_vec ** 2)
        if line_len2 == 0:
            return None, None

        # Project each junction point onto the perpendicular line
        t = np.dot(junction_line - p_start, line_vec) / line_len2
        t = np.clip(t, 0, 1)
        projection = p_start + np.outer(t, line_vec)

        # Find closest junction point
        distances = np.linalg.norm(junction_line - projection, axis=1)
        idx_min = np.argmin(distances)
        intersection = projection[idx_min]

        # Map intersection to closest index along profile array
        profile_idx = int(round(t[idx_min] * (profile_length - 1)))
        profile_idx = np.clip(profile_idx, 0, profile_length - 1)

        return intersection, profile_idx
#TODO
    def _plot_perpendicular_profiles(self, profiles):
        """
        Plot perpendicular lines for manual line with intersection markers.
        """
        if not profiles:
            print("No profiles to plot.")
            return

        # Delegate plotting to the shared perpendicular plotting utility which
        # draws the perpendicular lines on the main viewer (if ax/fig passed)
        # and opens a scrollable window with stacked SEM + log10(Current) plots.
        try:
            plot_perpendicular_profiles(profiles, ax=getattr(self, 'ax', None), fig=getattr(self, 'fig', None), debug=False)
        except Exception as e:
            print("Failed to open stacked perpendicular plots, falling back to simple plots:", e)
            # Fallback: draw simple twin-y plots as before
            if hasattr(self, 'ax') and self.ax:
                self.perp_lines = getattr(self, 'perp_lines', [])
                for prof in profiles:
                    p_start, p_end = prof["line_coords"]
                    line, = self.ax.plot([p_start[0], p_end[0]], [p_start[1], p_end[1]], 'r--', linewidth=1)
                    self.perp_lines.append(line)
                    if prof.get("intersection") is not None:
                        inter = prof["intersection"]
                        self.ax.scatter(inter[0], inter[1], color='lime', s=50, marker='x', zorder=20)

                if hasattr(self, 'fig') and self.fig:
                    self.fig.canvas.draw_idle()

            import tkinter as tk
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            win = tk.Toplevel()
            win.title("Perpendicular Profiles (Manual Line)")
            win.geometry("1200x800")
            win.resizable(True, True)
            win.lift()
            win.focus_force()

            canvas = tk.Canvas(win)
            scrollbar = tk.Scrollbar(win, orient="vertical", command=canvas.yview)
            scroll_frame = tk.Frame(canvas)
            scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
            canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")

            for prof in profiles:
                dist_um = prof["dist_um"]
                sem_vals = prof["sem"]
                cur_vals = prof["current"]

                sem_vals_norm = (sem_vals - np.min(sem_vals)) / (np.ptp(sem_vals) + 1e-12)

                # Create a 3-row stacked figure: SEM+current, log-current, derivative
                fig, (ax1, ax_log, ax_deriv) = plt.subplots(
                    3, 1, sharex=True, figsize=(6, 7), gridspec_kw={'height_ratios': [1, 1, 0.8]}
                )

                ax1.plot(dist_um, sem_vals_norm, color='tab:blue', linewidth=2, label='SEM (norm)')
                ax2 = ax1.twinx()
                ax2.plot(dist_um, cur_vals, color='tab:red', linewidth=1.5, label='Current (nA)')

                # Mark intersection point on axes
                if prof.get("intersection_idx") is not None:
                    idx = prof["intersection_idx"]
                    try:
                        ax1.scatter(dist_um[idx], sem_vals_norm[idx], color='lime', s=50, marker='x', zorder=10)
                        ax2.scatter(dist_um[idx], cur_vals[idx], color='lime', s=50, marker='x', zorder=10)
                    except Exception:
                        pass

                ax1.set_xlabel("Distance (µm)")
                ax1.set_ylabel("SEM Contrast (norm)", color='tab:blue')
                ax2.set_ylabel("Current (nA)", color='tab:red')
                ax1.set_title(f"Perpendicular {prof['id'] + 1}")
                ax1.legend(loc='upper left')

                # Middle: log-scale current
                cur = np.array(cur_vals)
                pos = cur[cur > 0]
                if pos.size > 0:
                    floor = max(np.min(pos) * 0.1, 1e-12)
                else:
                    floor = 1e-12
                cur_safe = np.maximum(cur, floor)
                ax_log.plot(dist_um, cur_safe, color='tab:orange', linewidth=1.5, label='Current (nA)')
                ax_log.set_yscale('log')

                # Bottom: numeric derivative dI/dx
                try:
                    deriv = np.gradient(cur, dist_um)
                    ax_deriv.plot(dist_um, deriv, color='tab:green', linewidth=1.2, label='dI/dx')
                    ax_deriv.fill_between(dist_um, deriv, 0, where=(deriv >= 0), interpolate=True, color='tab:green', alpha=0.25)
                    ax_deriv.fill_between(dist_um, deriv, 0, where=(deriv < 0), interpolate=True, color='tab:red', alpha=0.12)
                    if prof.get("intersection_idx") is not None:
                        try:
                            ax_deriv.scatter(dist_um[idx], deriv[idx], color='lime', s=40, marker='x', zorder=10)
                        except Exception:
                            pass
                    ax_deriv.set_ylabel('dI/dx (nA/µm)')
                    ax_deriv.grid(True, linestyle='--', alpha=0.3)
                    ax_deriv.legend(loc='upper left')
                except Exception:
                    ax_deriv.text(0.5, 0.5, 'dI/dx unavailable', ha='center', va='center')

                fig.tight_layout()

                from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
                plot_canvas = FigureCanvasTkAgg(fig, master=scroll_frame)
                widget = plot_canvas.get_tk_widget()
                widget.pack(pady=10)


    def _reset_overlays(self, event=None):
        """Remove main line and perpendicular lines, reset line coordinates."""
        # Remove main line
        if self.line is not None and hasattr(self.line, "remove"):
            self.line.remove()
            self.line = None

        # Reset coordinates
        self.line_coords = None
        self._clicks = []

        # Remove perpendicular lines if tracked
        if hasattr(self, 'perp_lines'):
            new_list = []
            for pl in self.perp_lines:
                try:
                    if hasattr(pl, "remove"):
                        pl.remove()
                except Exception as e:
                    print(f"Warning: could not remove line: {e}")
                    new_list.append(pl)  # keep if cannot remove
            self.perp_lines = new_list

        # Redraw image without overlays
        self._update_display()
        self.fig.canvas.draw_idle()

    def _detect_junction_button(self, event):
        """Callback for Junction Detection button using JunctionAnalyzer (Canny only)."""
        if self.line_coords is None:
            print("Please draw the rough line along the junction first.")
            return

        if not hasattr(self, 'manual_line_dense'):
            print("Manual line points not generated. Draw the line first.")
            return

        # Ask user for half-width
        width_um = ask_junction_width()
        if width_um is None:
            print("Junction detection canceled by user.")
            return

        # Convert to pixels
        half_width_px = int(np.round(width_um * 1e-6 / self.pixel_size))
        half_width_px = max(1, half_width_px)

        # Extract ROI from pixel map
        pixel_map = self.pixel_maps[self.index]
        roi, roi_coords = extract_line_rectangle(pixel_map, self.manual_line_dense, half_width_px)

        # Create analyzer (Canny only)
        analyzer = JunctionAnalyzer(pixel_size_m=self.pixel_size)

        # Try to build EBIC/current ROI if available
        roi_current = None
        try:
            if self.current_maps is not None and len(self.current_maps) > 1 and self.current_maps[1] is not None:
                current_map = self.current_maps[1]
                roi_current, _ = extract_line_rectangle(current_map, self.manual_line_dense, half_width_px)
        except Exception:
            roi_current = None

        # Use EBIC weight from slider if available, otherwise use default
        weight = getattr(self, 'ebic_weight', 10.0)
        print(f"Junction detection: using EBIC weight = {weight}")

        # Detect junction (returns a single result in a list)
        results = analyzer.detect(roi, self.manual_line_dense, roi_current=roi_current, weight_current=weight)

        if not results:
            print("Junction detection failed.")
            return

        # Persist last ROI and settings so slider changes can re-run detection
        self._last_roi = roi
        self._last_roi_current = roi_current
        self._last_manual_line_dense = self.manual_line_dense
        self._last_half_width_px = half_width_px

        # Visualize on the original pixel_map (image) and also draw the detected line on the viewer axes
        try:
            # Draw analyzer's visualization if available
            analyzer.visualize_results(pixel_map, self.manual_line_dense, results)
        except Exception:
            pass

        # Extract best detected coordinates and draw green line on viewer
        try:
            best_result = results[0]
            detected_coords = best_result[1]
            self.detected_junction_line = detected_coords
            self.detected_on_dataset_index = self.sweep_index if getattr(self, 'sweep_index', None) is not None else self.index
            # remove previous if present
            try:
                if hasattr(self, 'detected_line_obj') and self.detected_line_obj is not None:
                    try:
                        self.detected_line_obj.remove()
                    except Exception:
                        pass
            except Exception:
                pass

            # Map detected coords to currently displayed dataset coords
            try:
                mapped = self._get_displayed_detected_coords()
                if mapped is None:
                    mapped = self.detected_junction_line
            except Exception:
                mapped = self.detected_junction_line

            self.detected_line_obj = Line2D(mapped[:, 0], mapped[:, 1], color='green', linewidth=2)
            self.ax.add_line(self.detected_line_obj)
            if hasattr(self, 'fig') and self.fig is not None:
                self.fig.canvas.draw_idle()
        except Exception:
            pass
    def _observe_detected_junction_button(self, event=None):
        """Callback to sample SEM and EBIC along the detected junction and show plots."""
        try:
            if not hasattr(self, 'detected_junction_line') or self.detected_junction_line is None:
                print("No detected junction available. Run detection first.")
                return

            # Map detected junction coordinates to coordinates of the currently displayed dataset
            coords = self._get_displayed_detected_coords()
            if coords is None:
                coords = np.asarray(self.detected_junction_line)
            else:
                coords = np.asarray(coords)
            if coords.size == 0:
                print("Detected junction has no coordinates.")
                return

            # Ensure arrays of x and y
            xs = coords[:, 0]
            ys = coords[:, 1]

            # Use the currently displayed SEM/pixel data (respect sweep/dataset switches)
            sem_data = None
            try:
                sem_data = self.pixel_maps[self.index]
            except Exception:
                sem_data = None

            if sem_data is None:
                print("No SEM/pixel data available to sample.")
                return

            # Sample SEM and current along the detected coordinates from the currently displayed dataset
            sem_vals = map_coordinates(sem_data, [ys, xs], order=1, mode='nearest')

            cur_vals = None
            try:
                # Prefer processed current map if available (index 1), otherwise fallback to pixel_maps[1]
                cur_map = None
                if getattr(self, 'current_maps', None) is not None and len(self.current_maps) > 1 and self.current_maps[1] is not None:
                    cur_map = self.current_maps[1]
                elif len(self.pixel_maps) > 1 and self.pixel_maps[1] is not None:
                    cur_map = self.pixel_maps[1]

                if cur_map is not None:
                    cur_vals = map_coordinates(cur_map, [ys, xs], order=1, mode='nearest')
            except Exception:
                cur_vals = None

            # Compute distance along the detected line (in µm)
            diffs = np.sqrt(np.sum(np.diff(coords, axis=0) ** 2, axis=1))
            dists_px = np.concatenate(([0.0], np.cumsum(diffs)))
            dists_um = dists_px * self.pixel_size * 1e6

            # Normalize SEM for plotting
            sem_norm = (sem_vals - np.min(sem_vals)) / (np.ptp(sem_vals) + 1e-12)

            # Plot in a new figure (simple matplotlib window)
            fig, ax1 = plt.subplots(figsize=(8, 4))
            ax1.plot(dists_um, sem_norm, color='tab:blue', linewidth=2, label='SEM (norm)')
            ax1.set_xlabel('Distance along junction (µm)')
            ax1.set_ylabel('SEM (norm)', color='tab:blue')
            ax1.tick_params(axis='y', labelcolor='tab:blue')

            if cur_vals is not None:
                ax2 = ax1.twinx()
                ax2.plot(dists_um, cur_vals, color='tab:red', linewidth=1.5, label='Current (nA)')
                ax2.set_ylabel('Current (nA)', color='tab:red')
                ax2.tick_params(axis='y', labelcolor='tab:red')

            ax1.set_title(f"{self.sample_name} - Along detected junction")
            ax1.grid(True)
            fig.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Failed to observe detected junction: {e}")
    
    def _apply_detection_to_sweep(self, event=None):
        """Run junction detection and perpendicular profile calculation across all sweep datasets.

        This transforms the manual line into each dataset using computed integer shifts,
        extracts ROIs, runs JunctionAnalyzer.detect for each dataset with the current
        EBIC weight, and computes perpendicular profiles (if user provides parameters).
        Results are stored in self.sweep_detected_coords and self.sweep_perpendicular_profiles.
        """
        try:
            if not getattr(self, 'sweep_datasets', None):
                print("No sweep datasets available.")
                return

            # Determine the manual dense line to use
            manual_dense = None
            if getattr(self, '_last_manual_line_dense', None) is not None:
                manual_dense = self._last_manual_line_dense
            elif getattr(self, 'manual_line_dense', None) is not None:
                manual_dense = self.manual_line_dense
            else:
                print("No manual line available. Draw a line first.")
                return

            # Determine half-width in pixels
            half_width_px = getattr(self, '_last_half_width_px', None)
            if half_width_px is None:
                width_um = ask_junction_width()
                if width_um is None:
                    print("Operation canceled by user.")
                    return
                half_width_px = int(np.round(width_um * 1e-6 / self.pixel_size))
                half_width_px = max(1, half_width_px)

            # Ask perpendicular params (num lines and length)
            from tkinter import Tk, simpledialog
            root = Tk(); root.withdraw(); root.update()
            try:
                num_lines = simpledialog.askinteger("Input", "Number of perpendicular lines:", parent=root, minvalue=1, maxvalue=200)
                length_um = simpledialog.askfloat("Input", "Length of each perpendicular line (µm):", parent=root, minvalue=0.1, maxvalue=1e5)
            except Exception:
                root.destroy()
                print("Invalid input. Operation canceled.")
                return
            root.destroy()
            if num_lines is None or length_um is None:
                print("Operation canceled by user.")
                return

            n = len(self.sweep_datasets)
            detected_list = [None] * n
            perp_list = [None] * n

            ref_idx = int(self.sweep_index) if self.sweep_index is not None else 0

            # print(f"Starting sweep detection on {n} datasets (ref idx {ref_idx})...")
            for i, ds in enumerate(self.sweep_datasets):
                try:
                    pm_list = ds.get('pixel_maps', [])
                    cm_list = ds.get('current_maps', [])
                    if not pm_list:
                        # print(f"Dataset {i} missing pixel maps, skipping.")
                        continue
                    sem_data = pm_list[self.index] if (self.index is not None and self.index < len(pm_list)) else pm_list[0]

                    # map manual dense line to this dataset coordinates using shifts
                    if self.sweep_shifts is not None:
                        sx_i, sy_i = self.sweep_shifts[i]
                        sx_ref, sy_ref = self.sweep_shifts[ref_idx]
                        dx = sx_i - sx_ref
                        dy = sy_i - sy_ref
                        mapped_dense = manual_dense + np.array([dx, dy])
                    else:
                        mapped_dense = manual_dense.copy()

                    # Extract ROI for SEM and current (if present)
                    roi, _ = extract_line_rectangle(sem_data, mapped_dense, half_width_px)
                    roi_current = None
                    try:
                        if cm_list and len(cm_list) > 1 and cm_list[1] is not None:
                            roi_current, _ = extract_line_rectangle(cm_list[1], mapped_dense, half_width_px)
                    except Exception:
                        roi_current = None

                    analyzer = JunctionAnalyzer(pixel_size_m=ds.get('pixel_size', self.pixel_size))
                    results = analyzer.detect(roi, mapped_dense, roi_current=roi_current, weight_current=self.ebic_weight, debug=False)
                    if results:
                        best = results[0]
                        # best is expected to be (label, coords, metrics)
                        if isinstance(best, (list, tuple)) and len(best) >= 2:
                            detected_coords = best[1]
                        else:
                            detected_coords = np.asarray(best)
                        detected_list[i] = np.asarray(detected_coords)
                        # Compute perpendicular profiles using detected junction as intersection reference
                        try:
                            profiles = calculate_perpendicular_profiles(mapped_dense, int(num_lines), float(length_um), sem_data, cm_list[1] if (cm_list and len(cm_list) > 1) else None, pixel_size_m=ds.get('pixel_size', self.pixel_size), detected_junction=detected_coords, source_name=ds.get('sample_name'))
                            perp_list[i] = profiles
                        except Exception as e:
                            pass  # print(f"Failed to compute perpendiculars for dataset {i}: {e}")
                    else:
                        pass  # print(f"No detection result for dataset {i}.")
                except Exception as e:
                    pass  # print(f"Error processing dataset {i}: {e}")

            # Persist results
            self.sweep_detected_coords = detected_list
            self.sweep_perpendicular_profiles = perp_list

            # If current displayed dataset now has a detection, update the viewer's detected_junction_line
            try:
                cur_idx = int(self.sweep_index) if self.sweep_index is not None else 0
                if 0 <= cur_idx < len(detected_list) and detected_list[cur_idx] is not None:
                    self.detected_junction_line = detected_list[cur_idx]
                    self.detected_on_dataset_index = cur_idx
                # redraw
                self._update_display()
            except Exception:
                pass

            # Ensure the viewer.perpendicular_profiles is set to the profiles for the currently displayed dataset
            try:
                cur_idx = int(self.sweep_index) if self.sweep_index is not None else 0
                if 0 <= cur_idx < len(perp_list) and perp_list[cur_idx] is not None:
                    # store the current dataset's perpendiculars in the viewer attribute so fit functions find them
                    self.perpendicular_profiles = perp_list[cur_idx]
                else:
                    # fallback to empty list
                    self.perpendicular_profiles = []
            except Exception:
                self.perpendicular_profiles = []

            # print("Sweep detection finished.")
            # Optionally open debug view automatically if enabled
            try:
                if getattr(self, 'debug_mode', False):
                    self._debug_show_sweep()
            except Exception:
                pass
        except Exception as e:
            print(f"Failed to run sweep detection: {e}")
    
    def _debug_show_sweep(self, event=None):
        """Open a figure showing each sweep dataset image with detected junction and perpendiculars (if available)."""
        try:
            if not getattr(self, 'sweep_datasets', None):
                print("No sweep datasets available for debug display.")
                return

            n = len(self.sweep_datasets)
            if n == 0:
                print("Sweep is empty.")
                return

            # layout
            ncols = min(4, n)
            nrows = int(np.ceil(n / ncols))
            fig = plt.figure(figsize=(4 * ncols, 3 * nrows))

            ref_idx = int(self.sweep_index) if self.sweep_index is not None else 0
            for i, ds in enumerate(self.sweep_datasets):
                ax = fig.add_subplot(nrows, ncols, i + 1)
                pm = ds.get('pixel_maps', [])
                if not pm:
                    ax.set_title(f"Dataset {i}: no pixel map")
                    continue
                img = pm[0]
                ax.imshow(img, cmap='gray', extent=[0, img.shape[1], img.shape[0], 0])
                ax.set_xticks([]); ax.set_yticks([])
                title = ds.get('sample_name', f'dataset_{i}')
                ax.set_title(f"{i}: {title}")

                # Draw the computed integer shift vector (relative to reference dataset)
                try:
                    if getattr(self, 'sweep_shifts', None) is not None and ref_idx is not None:
                        sx_ref, sy_ref = self.sweep_shifts[ref_idx]
                        sx_i, sy_i = self.sweep_shifts[i]
                        ddx = sx_i - sx_ref
                        ddy = sy_i - sy_ref
                        # place the arrow near top-left of the image for visibility
                        h, w = img.shape[0], img.shape[1]
                        anchor_x = max(5, int(0.05 * w))
                        anchor_y = max(5, int(0.05 * h))
                        end_x = anchor_x + ddx
                        end_y = anchor_y + ddy
                        # draw arrow and text in yellow for contrast
                        ax.annotate('', xy=(end_x, end_y), xytext=(anchor_x, anchor_y),
                                    arrowprops=dict(arrowstyle='->', color='yellow', linewidth=1.5))
                        ax.text(anchor_x, anchor_y + 12, f"shift: {ddx},{ddy}", color='yellow', fontsize=8, va='top')
                except Exception:
                    pass

                # get detected coords for this dataset if available
                det_coords = None
                try:
                    if getattr(self, 'sweep_detected_coords', None) is not None:
                        det_coords = self.sweep_detected_coords[i]
                except Exception:
                    det_coords = None

                # fallback: map global detected_junction_line to this dataset
                if det_coords is None and getattr(self, 'detected_junction_line', None) is not None:
                    try:
                        # map from detected_on_dataset_index to i
                        src_idx = getattr(self, 'detected_on_dataset_index', self.sweep_index)
                        if self.sweep_shifts is not None and src_idx is not None:
                            sx_i, sy_i = self.sweep_shifts[i]
                            sx_s, sy_s = self.sweep_shifts[src_idx]
                            dx = sx_i - sx_s
                            dy = sy_i - sy_s
                            det_coords = np.asarray(self.detected_junction_line) + np.array([dx, dy])
                        else:
                            det_coords = np.asarray(self.detected_junction_line)
                    except Exception:
                        det_coords = None

                if det_coords is not None:
                    try:
                        ax.plot(det_coords[:, 0], det_coords[:, 1], color='lime', linewidth=2)
                    except Exception:
                        pass

                # draw perpendiculars if available for this dataset
                try:
                    if getattr(self, 'sweep_perpendicular_profiles', None) is not None:
                        profs = self.sweep_perpendicular_profiles[i]
                        if profs:
                            for prof in profs:
                                p_start, p_end = prof.get('line_coords', (None, None))
                                if p_start is not None and p_end is not None:
                                    ax.plot([p_start[0], p_end[0]], [p_start[1], p_end[1]], 'r--', linewidth=0.8)
                                inter = prof.get('intersection')
                                if inter is not None:
                                    ax.scatter(inter[0], inter[1], color='magenta', s=20, marker='x')
                except Exception:
                    pass

            fig.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Failed to open debug sweep view: {e}")

    def _toggle_debug_mode(self, event=None):
        """Toggle debug mode on/off and update button label."""
        try:
            self.debug_mode = not getattr(self, 'debug_mode', False)
            try:
                if hasattr(self, 'b_debug_mode') and self.b_debug_mode is not None:
                    self.b_debug_mode.label.set_text('Debug: ON' if self.debug_mode else 'Debug: OFF')
            except Exception:
                pass
            print(f"Debug mode {'ON' if self.debug_mode else 'OFF'}")
        except Exception:
            pass
    
    def _fit_profiles(self, event=None):
        # If sweep is loaded, run fitting for every dataset's perpendicular profiles
        try:
            if getattr(self, 'sweep_datasets', None) and getattr(self, 'sweep_perpendicular_profiles', None):
                results_all = []
                orig_profiles = getattr(self, 'perpendicular_profiles', None)
                for i, profs in enumerate(self.sweep_perpendicular_profiles):
                    if not profs:
                        print(f"Skipping dataset {i} (no perpendiculars)")
                        continue
                    print(f"Fitting exponential profiles for dataset {i} ({self.sweep_datasets[i].get('sample_name',i)})...")
                    # set current profiles so helper finds them
                    self.perpendicular_profiles = profs
                    try:
                        res = fit_perpendicular_profiles(self)
                        results_all.append((i, res))
                    except Exception as e:
                        print(f"Failed to fit profiles for dataset {i}: {e}")
                # restore original profiles
                self.perpendicular_profiles = orig_profiles
                return results_all
        except Exception:
            pass

        # fallback: fit on current viewer perpendicular_profiles
        return fit_perpendicular_profiles(self)

    def _fit_profiles_linear(self, event=None):
        try:
            if getattr(self, 'sweep_datasets', None) and getattr(self, 'sweep_perpendicular_profiles', None):
                results_all = []
                orig_profiles = getattr(self, 'perpendicular_profiles', None)
                for i, profs in enumerate(self.sweep_perpendicular_profiles):
                    if not profs:
                        print(f"Skipping dataset {i} (no perpendiculars)")
                        continue
                    print(f"Fitting linear profiles for dataset {i} ({self.sweep_datasets[i].get('sample_name',i)})...")
                    self.perpendicular_profiles = profs
                    try:
                        res = fit_perpendicular_profiles_linear(self)
                        results_all.append((i, res))
                    except Exception as e:
                        print(f"Failed linear fit for dataset {i}: {e}")
                self.perpendicular_profiles = orig_profiles
                return results_all
        except Exception:
            pass

        return fit_perpendicular_profiles_linear(self)

    def _update_detected_junction_with_weight(self):
        """Re-run junction detection on the last ROI using the current EBIC weight
        and update the green detected line on the main viewer axes.
        """
        try:
            if self._last_roi is None or self._last_manual_line_dense is None:
                return

            analyzer = JunctionAnalyzer(pixel_size_m=self.pixel_size)
            results = analyzer.detect(self._last_roi, self._last_manual_line_dense,
                                      roi_current=self._last_roi_current,
                                      weight_current=self.ebic_weight)

            if not results:
                # remove previous visualization if it exists
                try:
                    if hasattr(self, 'detected_line_obj') and self.detected_line_obj is not None:
                        try:
                            self.detected_line_obj.remove()
                        except Exception:
                            pass
                        self.detected_line_obj = None
                        self.detected_junction_line = None
                        if hasattr(self, 'fig') and self.fig is not None:
                            self.fig.canvas.draw_idle()
                except Exception:
                    pass
                return

            best_result = results[0]
            detected_coords = best_result[1]
            self.detected_junction_line = detected_coords
            self.detected_on_dataset_index = self.sweep_index if getattr(self, 'sweep_index', None) is not None else self.index

            # remove previous detected line visualization if present
            try:
                if hasattr(self, 'detected_line_obj') and self.detected_line_obj is not None:
                    try:
                        self.detected_line_obj.remove()
                    except Exception:
                        pass
            except Exception:
                pass

            try:
                mapped = self._get_displayed_detected_coords()
                if mapped is None:
                    mapped = self.detected_junction_line
            except Exception:
                mapped = self.detected_junction_line

            self.detected_line_obj = Line2D(mapped[:, 0], mapped[:, 1], color='green', linewidth=2)
            try:
                self.ax.add_line(self.detected_line_obj)
            except Exception:
                pass

            if hasattr(self, 'fig') and self.fig is not None:
                self.fig.canvas.draw_idle()
        except Exception as e:
            print(f"Failed to update detected junction with new weight: {e}")
