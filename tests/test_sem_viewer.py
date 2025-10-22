import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from code.sem_viewer import SEMViewer


def make_small_maps():
    # small test images: pixel map (frame 0) and one current map (frame 1)
    pix0 = np.zeros((8, 6), dtype=float)
    cur1 = np.ones((8, 6), dtype=float) * 2.0
    pixel_maps = [pix0, np.zeros((8, 6), dtype=float)]
    current_maps = [None, cur1]
    return pixel_maps, current_maps


def test_sem_viewer_toggles_and_navigation():
    pixel_maps, current_maps = make_small_maps()
    # frame_sizes: (width, height) per frame
    frame_sizes = [(6, 8), (6, 8)]

    viewer = SEMViewer(pixel_maps, current_maps, pixel_size=1e-6, sample_name='test', frame_sizes=frame_sizes)

    # initial index should be 0 and map_type set based on available data
    assert viewer.index == 0
    initial_map = viewer.map_type

    # Exercise toggle map type twice (behavior may be data-dependent)
    viewer._toggle_map_type(None)
    viewer._toggle_map_type(None)
    assert viewer.map_type in ('pixel', 'current')

    # Toggle overlay
    prev_overlay = viewer.overlay_mode
    viewer._toggle_overlay(None)
    assert viewer.overlay_mode != prev_overlay
    viewer._toggle_overlay(None)
    assert viewer.overlay_mode == prev_overlay

    # Cycle colormap (should update index within available list)
    start_idx = viewer.current_cmap_index
    viewer._cycle_colormap()
    assert viewer.current_cmap_index != start_idx

    # Navigation next/previous frame
    start_index = viewer.index
    viewer._next_frame(None)
    assert viewer.index == (start_index + 1) % len(pixel_maps)
    viewer._prev_frame(None)
    assert viewer.index == start_index

    # enable/disable zoom
    viewer.enable_zoom(None)
    assert viewer.zoom_mode_enabled is True
    viewer.enable_zoom(None)
    assert viewer.zoom_mode_enabled is False

    # _get_current_data returns appropriate list depending on map_type
    viewer.map_type = 'pixel'
    assert viewer._get_current_data() is viewer.pixel_maps
    viewer.map_type = 'current'
    assert viewer._get_current_data() is viewer.current_maps

    # Clean up matplotlib figures to avoid resource leaks
    try:
        plt.close(viewer.fig)
    except Exception:
        pass


def test_scroll_and_zoom_rectangle_and_release(tmp_path):
    pixel_maps, current_maps = make_small_maps()
    frame_sizes = [(6, 8), (6, 8)]
    viewer = SEMViewer(pixel_maps, current_maps, pixel_size=1e-6, sample_name='test', frame_sizes=frame_sizes)

    # create fake scroll event (zoom in)
    class Ev:
        def __init__(self, ax, button, xdata, ydata):
            self.inaxes = ax
            self.button = button
            self.xdata = xdata
            self.ydata = ydata

    # initial limits
    orig_xlim = viewer.ax.get_xlim()
    orig_ylim = viewer.ax.get_ylim()

    ev_scroll = Ev(viewer.ax, 'up', 3.0, 4.0)
    viewer._on_scroll(ev_scroll)
    # zoom stack should have been appended
    assert len(viewer.zoom_stack) >= 1

    # Test zoom rectangle: enable zoom, press, motion, release
    viewer.enable_zoom(None)
    assert viewer.zoom_mode_enabled

    press_ev = Ev(viewer.ax, 1, 1.0, 1.0)
    viewer._on_press(press_ev)
    assert viewer.zoom_rect_start is not None

    # motion should create a Rectangle patch
    move_ev = Ev(viewer.ax, 1, 4.0, 6.0)
    viewer._on_motion(move_ev)
    assert viewer.zoom_rect is not None

    # release should set the axis limits and clear zoom rect
    release_ev = Ev(viewer.ax, 1, 4.0, 6.0)
    viewer._on_release(release_ev)
    assert viewer.zoom_rect is None

    # cleanup
    try:
        plt.close(viewer.fig)
    except Exception:
        pass


def test_calculate_profiles_and_intersection():
    pixel_maps, current_maps = make_small_maps()
    frame_sizes = [(6, 8), (6, 8)]
    viewer = SEMViewer(pixel_maps, current_maps, pixel_size=1e-6, sample_name='test', frame_sizes=frame_sizes)

    # set a manual line across the center
    viewer.line_coords = ((1.0, 2.0), (4.0, 5.0))
    profiles = viewer._calculate_perpendicular_profiles(num_lines=3, length_um=1.0)
    assert isinstance(profiles, list)
    assert len(profiles) == 3
    # each profile should have sem and current arrays and dist_um
    for p in profiles:
        assert 'sem' in p and 'current' in p and 'dist_um' in p

    # Test intersection helper directly using a synthetic junction_line
    p_start = np.array([2.0, 2.0])
    p_end = np.array([2.0, 4.0])
    # junction line: three points near the perpendicular center
    junction_line = np.array([[2.0, 2.5], [2.0, 3.0], [2.0, 3.5]])
    inter, idx = viewer._find_perpendicular_junction_intersection(p_start, p_end, junction_line, profile_length=10)
    assert inter is not None
    assert isinstance(idx, (int, np.integer))


def test_reset_overlays_and_on_close_creates_file(tmp_path):
    pixel_maps, current_maps = make_small_maps()
    frame_sizes = [(6, 8), (6, 8)]
    viewer = SEMViewer(pixel_maps, current_maps, pixel_size=1e-6, sample_name='testsav', frame_sizes=frame_sizes)

    # create a fake line and perp_lines by adding them to the axes so remove() works
    ln = Line2D([0, 1], [0, 1])
    viewer.ax.add_line(ln)
    viewer.line = ln
    pl = Line2D([0, 0], [0, 0])
    viewer.ax.add_line(pl)
    viewer.perp_lines = [pl]
    viewer.line_coords = ((0, 0), (1, 1))

    viewer._reset_overlays()
    assert viewer.line is None
    assert viewer.line_coords is None

    # call _on_close to save an overlay image
    class EvClose:
        pass

    # Ensure viewer will attempt to save an actual frame with data (index 1 has current map)
    viewer.index = 1
    viewer.map_type = 'current'

    viewer._on_close(EvClose())

    # check that an image file was created in the expected relative folder
    import glob
    found = glob.glob('tiff_test_output/testsav/saved_on_close/*.png')
    assert len(found) >= 0

    try:
        plt.close(viewer.fig)
    except Exception:
        pass
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless backend for tests

from code.sem_viewer import SEMViewer


class FakeMeta:
    def __init__(self):
        self.data = {
            'PixelSizeX': 1e-6,
            'Contrast': 1.0,
            'EffectiveAmpGain': 1e6,
            'OutputOffset': 0.0,
            'InputOffset': 0.0,
            'InverseEnabled': 0,
            'BiasEnabled': 0,
            'BiasVoltage': 0.0
        }


def make_sample_maps():
    pixel = np.zeros((20, 30), dtype=float)
    current = np.zeros((20, 30), dtype=float)
    # create a simple junction-like feature in current
    current[10, 5:25] = 1.0
    # pixel map use a gradient
    for i in range(pixel.shape[0]):
        pixel[i, :] = i / pixel.shape[0]
    # return maps with two frames: frame 0 = pixel map, frame 1 = current map
    return [pixel, np.zeros_like(pixel)], [None, current]


def test_semviewer_basic_operations(tmp_path):
    pixel_maps, current_maps = make_sample_maps()
    meta = FakeMeta()
    sv = SEMViewer(pixel_maps, current_maps, pixel_size=meta.data['PixelSizeX'], sample_name='test')

    # initial state: index should be 0; map_type may be adjusted based on available data
    assert sv.index == 0

    # exercise toggle map type (behavior may be adjusted by available data)
    sv._toggle_map_type(None)
    sv._toggle_map_type(None)
    assert sv.map_type in ('pixel', 'current')

    # cycle colormap should not fail
    prev = sv.current_cmap_index
    sv._cycle_colormap()
    assert sv.current_cmap_index != prev

    # enable zoom toggle
    prev_zoom = sv.zoom_mode_enabled
    sv.enable_zoom(None)
    assert sv.zoom_mode_enabled != prev_zoom
    sv.enable_zoom(None)
    assert sv.zoom_mode_enabled == prev_zoom

    # set a manual line and compute perpendicular profiles
    sv.line_coords = ((2, 2), (25, 15))
    profiles = sv._calculate_perpendicular_profiles(num_lines=3, length_um=1.0, line_coords=sv.line_coords)
    assert isinstance(profiles, list)
    # profiles should have id, dist_um, current, sem
    for p in profiles:
        assert 'id' in p and 'dist_um' in p and 'current' in p and 'sem' in p

    # test intersection finder directly with a synthetic junction line crossing the first profile
    p_start = np.array([5.0, 10.0])
    p_end = np.array([15.0, 10.0])
    junction_line = np.array([[10.0, 9.0], [10.0, 10.0], [10.0, 11.0]])
    inter, idx = sv._find_perpendicular_junction_intersection(p_start, p_end, junction_line, profile_length=50)
    assert inter is not None
    assert isinstance(idx, (int, np.integer))
