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
from matplotlib.widgets import Button
import matplotlib.patches as patches
from scipy.ndimage import map_coordinates
from scipy.stats import pearsonr
from .Junction_Analyser import JunctionAnalyzer
from .perpendicular import calculate_perpendicular_profiles, plot_perpendicular_profiles
from .helper_gui import fit_perpendicular_profiles
from .helper_gui import ask_junction_width, ask_junction_weight, draw_scalebar, safe_remove_colorbar
from .ROI_extractor import extract_line_rectangle

# ==================== SEM VIEWER ====================
class SEMViewer:
    def __init__(self, pixel_maps, current_maps, pixel_size, sample_name, frame_sizes=None, dpi=100):
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

    def _init_ui(self):
        # Fixed image axes: occupy full height, leave space for buttons on the right
        img_width_frac = 0.85  # fraction of figure width for image
        img_height_frac = 1.0  # full height
        self.ax.set_position([0, 0, img_width_frac, img_height_frac])

        # Button layout parameters
        button_w = 0.10
        button_h = 0.05
        spacing = 0.02
        x_pos = img_width_frac + spacing  # buttons start to the right of image
        y_start = 0.80  # top of first button (fraction of figure height)
        y = y_start

        # Overlay button
        ax_overlay = self.fig.add_axes([x_pos, y, button_w, button_h])
        self.b_overlay = Button(ax_overlay, 'Overlay')
        self.b_overlay.on_clicked(self._toggle_overlay)
        y -= (button_h + spacing)

        # Toggle Map button
        ax_toggle = self.fig.add_axes([x_pos, y, button_w, button_h])
        self.b_toggle = Button(ax_toggle, 'Toggle Map')
        self.b_toggle.on_clicked(self._toggle_map_type)
        y -= (button_h + spacing)

        # Previous Frame button
        ax_prev = self.fig.add_axes([x_pos, y, button_w, button_h])
        self.b_prev = Button(ax_prev, 'Previous')
        self.b_prev.on_clicked(self._prev_frame)
        y -= (button_h + spacing)

        # Next Frame button
        ax_next = self.fig.add_axes([x_pos, y, button_w, button_h])
        self.b_next = Button(ax_next, 'Next')
        self.b_next.on_clicked(self._next_frame)
        y -= (button_h + spacing)

        # Palette button
        ax_palette = self.fig.add_axes([x_pos, y, button_w, button_h])
        self.b_palette = Button(ax_palette, 'Palette')
        self.b_palette.on_clicked(self._cycle_colormap)
        y -= (button_h + spacing)

        # Perpendiculars button
        ax_perp = self.fig.add_axes([x_pos, y, button_w, button_h])
        self.b_perp = Button(ax_perp, 'Perpendiculars')
        self.b_perp.on_clicked(self._show_perpendicular_input)
        y -= (button_h + spacing)

        # Reset Zoom button
        ax_reset = self.fig.add_axes([x_pos, y, button_w, button_h])
        self.b_reset = Button(ax_reset, 'Reset Zoom')
        self.b_reset.on_clicked(self._reset_zoom)
        y -= (button_h + spacing)

        # Reset Lines button
        ax_reset_overlays = self.fig.add_axes([x_pos, y, button_w, button_h])
        self.b_reset_overlays = Button(ax_reset_overlays, 'Reset Lines')
        self.b_reset_overlays.on_clicked(self._reset_overlays)
        y -= (button_h + spacing)

        # Zoom button at the bottom
        ax_zoom = self.fig.add_axes([x_pos, 0.01, button_w, button_h])
        self.zoom_btn = Button(ax_zoom, 'Zoom')
        self.zoom_btn.on_clicked(self.enable_zoom)

        # Fit Profiles button
        ax_fit = self.fig.add_axes([x_pos, y, button_w, button_h])
        self.b_fit = Button(ax_fit, 'Fit Profiles')
        self.b_fit.on_clicked(self._fit_profiles)
        y -= (button_h + spacing)

        # --- Junction Detection button ---
        ax_junc = self.fig.add_axes([x_pos, y, button_w, button_h])
        self.b_junction = Button(ax_junc, 'Junction Detect')
        self.b_junction.on_clicked(self._detect_junction_button)
        y -= (button_h + spacing)


        # Connect close event
        self.fig.canvas.mpl_connect('close_event', self._on_close)

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
                    pixel = self.pixel_maps[0]
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

            # Automatically run junction detection
            try:
                width_um = ask_junction_width()
                if width_um is not None:
                    half_width_px = int(np.round(width_um * 1e-6 / self.pixel_size))
                    half_width_px = max(1, half_width_px)
                    pixel_map = self.pixel_maps[self.index]
                    roi, roi_coords = extract_line_rectangle(pixel_map,
                                                                  self.manual_line_dense,
                                                                  half_width_px)

                    # Try to build EBIC/current ROI if available
                    roi_current = None
                    try:
                        if self.current_maps is not None and len(self.current_maps) > 1 and self.current_maps[1] is not None:
                            current_map = self.current_maps[1]
                            roi_current, _ = extract_line_rectangle(current_map, self.manual_line_dense, half_width_px)
                    except Exception:
                        roi_current = None

                    # Ask user for EBIC weight (1.0 by default)
                    try:
                        weight = ask_junction_weight()
                        if weight is None:
                            weight = 10.0
                    except Exception:
                        weight = 10.0
                    print(f"Junction detection: using EBIC weight = {weight}")

                    analyzer = JunctionAnalyzer(pixel_size_m=self.pixel_size)
                    results = analyzer.detect(roi, self.manual_line_dense, roi_current=roi_current, weight_current=weight)

                    if results:
                        # Get the best result (which is the only one returned by Canny)
                        best_result = results[0]
                        detected_coords = best_result[1]

                        # Store the detected line and its visual object as class attributes
                        self.detected_junction_line = detected_coords
                        self.detected_line_obj = Line2D(
                            self.detected_junction_line[:, 0],
                            self.detected_junction_line[:, 1],
                            color='green',
                            linewidth=2
                        )
                        self.ax.add_line(self.detected_line_obj)

                    else:
                        print("Junction detection failed.")
            except Exception as e:
                print(f"Auto junction detection error: {e}")


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
            plt.close(fig)
            return

        cmap = 'gray' if self.map_type == 'pixel' else self.current_colormaps[self.current_cmap_index]
        ax.imshow(data, cmap=cmap)
        if self.line is not None:
            ax.add_line(Line2D(self.line.get_xdata(), self.line.get_ydata(), color='red'))

        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(img_save_path)
        plt.close(fig)
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

        # === Calculate perpendicular profiles ===
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

        sem_data = self.pixel_maps[0]
        current_data = self.current_maps[1]
        if sem_data is None or current_data is None:
            print("Missing SEM or current data.")
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

        # --- Draw perpendicular lines on main viewer ---
        if hasattr(self, 'ax') and self.ax:
            self.perp_lines = getattr(self, 'perp_lines', [])
            for prof in profiles:
                p_start, p_end = prof["line_coords"]
                line, = self.ax.plot([p_start[0], p_end[0]], [p_start[1], p_end[1]], 'r--', linewidth=1)
                self.perp_lines.append(line)
                # Draw intersection if available
                if prof.get("intersection") is not None:
                    inter = prof["intersection"]
                    self.ax.scatter(inter[0], inter[1], color='lime', s=50, marker='x', zorder=20)

            if hasattr(self, 'fig') and self.fig:
                self.fig.canvas.draw_idle()

        # --- Scrollable profile plot window ---
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

            fig, ax1 = plt.subplots(figsize=(6, 3))
            ax1.plot(dist_um, sem_vals_norm, color='tab:blue', linewidth=2, label='SEM (norm)')
            ax2 = ax1.twinx()
            ax2.plot(dist_um, cur_vals, color='tab:red', linewidth=1.5, label='Current (nA)')

            # Mark intersection point
            if prof.get("intersection_idx") is not None:
                idx = prof["intersection_idx"]
                ax1.scatter(dist_um[idx], sem_vals_norm[idx], color='lime', s=50, marker='x', zorder=10)
                ax2.scatter(dist_um[idx], cur_vals[idx], color='lime', s=50, marker='x', zorder=10)

            ax1.set_xlabel("Distance (µm)")
            ax1.set_ylabel("SEM Contrast (norm)", color='tab:blue')
            ax2.set_ylabel("Current (nA)", color='tab:red')
            ax1.set_title(f"Perpendicular {prof['id'] + 1}")
            ax1.legend(loc='upper left')
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

        # Ask user for EBIC weight
        try:
            weight = ask_junction_weight()
            if weight is None:
                weight = 10.0
        except Exception:
            weight = 10.0
        print(f"Junction detection: using EBIC weight = {weight}")

        # Detect junction (returns a single result in a list)
        results = analyzer.detect(roi, self.manual_line_dense, roi_current=roi_current, weight_current=weight)

        if not results:
            print("Junction detection failed.")
            return

        # Visualize on the original pixel_map (image)
        analyzer.visualize_results(pixel_map, self.manual_line_dense, results)
    
    def _fit_profiles(self, event=None):
        fit_perpendicular_profiles(self)
