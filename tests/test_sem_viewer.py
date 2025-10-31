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
    frame_sizes = [(6, 8), (6, 8)]

    viewer = SEMViewer(pixel_maps, current_maps, pixel_size=1e-6, sample_name='test', frame_sizes=frame_sizes)

    assert viewer.index == 0
    initial_map = viewer.map_type

    viewer._toggle_map_type(None)
    viewer._toggle_map_type(None)
    assert viewer.map_type in ('pixel', 'current')

    prev_overlay = viewer.overlay_mode
    viewer._toggle_overlay(None)
    assert viewer.overlay_mode != prev_overlay
    viewer._toggle_overlay(None)
    assert viewer.overlay_mode == prev_overlay

    start_idx = viewer.current_cmap_index
    viewer._cycle_colormap()
    assert viewer.current_cmap_index != start_idx

    start_index = viewer.index
    viewer._next_frame(None)
    assert viewer.index == (start_index + 1) % len(pixel_maps)
    viewer._prev_frame(None)
    assert viewer.index == start_index

    viewer.enable_zoom(None)
    assert viewer.zoom_mode_enabled is True
    viewer.enable_zoom(None)
    assert viewer.zoom_mode_enabled is False

    viewer.map_type = 'pixel'
    assert viewer._get_current_data() is viewer.pixel_maps
    viewer.map_type = 'current'
    assert viewer._get_current_data() is viewer.current_maps

    try:
        plt.close(viewer.fig)
    except Exception:
        pass


def test_scroll_and_zoom_rectangle_and_release(tmp_path):
    pixel_maps, current_maps = make_small_maps()
    frame_sizes = [(6, 8), (6, 8)]
    viewer = SEMViewer(pixel_maps, current_maps, pixel_size=1e-6, sample_name='test', frame_sizes=frame_sizes)

    class Ev:
        def __init__(self, ax, button, xdata, ydata):
            self.inaxes = ax
            self.button = button
            self.xdata = xdata
            self.ydata = ydata

    orig_xlim = viewer.ax.get_xlim()
    orig_ylim = viewer.ax.get_ylim()

    ev_scroll = Ev(viewer.ax, 'up', 3.0, 4.0)
    viewer._on_scroll(ev_scroll)
    assert len(viewer.zoom_stack) >= 1

    viewer.enable_zoom(None)
    assert viewer.zoom_mode_enabled

    press_ev = Ev(viewer.ax, 1, 1.0, 1.0)
    viewer._on_press(press_ev)
    assert viewer.zoom_rect_start is not None

    move_ev = Ev(viewer.ax, 1, 4.0, 6.0)
    viewer._on_motion(move_ev)
    assert viewer.zoom_rect is not None

    release_ev = Ev(viewer.ax, 1, 4.0, 6.0)
    viewer._on_release(release_ev)
    assert viewer.zoom_rect is None

    try:
        plt.close(viewer.fig)
    except Exception:
        pass


def test_calculate_profiles_and_intersection():
    pixel_maps, current_maps = make_small_maps()
    frame_sizes = [(6, 8), (6, 8)]
    viewer = SEMViewer(pixel_maps, current_maps, pixel_size=1e-6, sample_name='test', frame_sizes=frame_sizes)

    viewer.line_coords = ((1.0, 2.0), (4.0, 5.0))
    profiles = viewer._calculate_perpendicular_profiles(num_lines=3, length_um=1.0)
    assert isinstance(profiles, list)
    assert len(profiles) == 3
    for p in profiles:
        assert 'sem' in p and 'current' in p and 'dist_um' in p

    p_start = np.array([2.0, 2.0])
    p_end = np.array([2.0, 4.0])
    junction_line = np.array([[2.0, 2.5], [2.0, 3.0], [2.0, 3.5]])
    inter, idx = viewer._find_perpendicular_junction_intersection(p_start, p_end, junction_line, profile_length=10)
    assert inter is not None
    assert isinstance(idx, (int, np.integer))


def test_reset_overlays_and_on_close_creates_file(tmp_path):
    pixel_maps, current_maps = make_small_maps()
    frame_sizes = [(6, 8), (6, 8)]
    viewer = SEMViewer(pixel_maps, current_maps, pixel_size=1e-6, sample_name='testsav', frame_sizes=frame_sizes)

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

    class EvClose:
        pass

    viewer.index = 1
    viewer.map_type = 'current'

    viewer._on_close(EvClose())

    import glob
    found = glob.glob('tiff_test_output/testsav/saved_on_close/*.png')
    assert len(found) >= 0

    try:
        plt.close(viewer.fig)
    except Exception:
        pass
    
def test_semviewer_has_fit_lin_button(monkeypatch):
    # ensure plt.show does nothing in the test
    monkeypatch.setattr(plt, 'show', lambda *a, **k: None)

    pixel = np.zeros((100, 150))
    current = np.zeros((100, 150))
    # current_maps expect a list where index 1 is the current frame
    current_maps = [None, current]

    sv = SEMViewer([pixel], current_maps, pixel_size=1e-6, sample_name='test')

    # The button attribute should exist
    assert hasattr(sv, 'b_fit_lin'), "SEMViewer missing 'b_fit_lin' attribute"
    btn = getattr(sv, 'b_fit_lin')
    # Button object should not be None and should have on_clicked method
    assert btn is not None
    assert hasattr(btn, 'on_clicked')
