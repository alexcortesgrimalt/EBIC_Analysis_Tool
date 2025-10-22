import numpy as np
import pytest

from code.Junction_Analyser import JunctionAnalyzer


def make_edge_roi(height=20, width=30, edge_row=None):
    if edge_row is None:
        edge_row = height // 2
    roi = np.zeros((height, width), dtype=np.uint8)
    roi[edge_row:] = 255
    return roi


def test_apply_preprocessing_and_detect_canny_basic():
    roi = make_edge_roi(height=20, width=30, edge_row=10)
    ja = JunctionAnalyzer(pixel_size_m=1e-6)

    filtered = ja._apply_preprocessing_filter(roi)
    assert filtered.shape == roi.shape
    assert filtered.dtype == np.uint8

    detected = ja._detect_junction_canny(roi)
    # detected should have shape (W, 2) and first column equals column indices
    assert detected.shape == (roi.shape[1], 2)
    cols = detected[:, 0]
    assert np.allclose(cols, np.arange(roi.shape[1]))


def test_spline_and_line_postprocessing_shapes():
    ja = JunctionAnalyzer(pixel_size_m=1e-6)
    x = np.linspace(0, 10, 20)
    y = np.linspace(5, 15, 20)
    detected_coords = np.column_stack([x, y])

    smooth = ja._apply_spline_postprocessing(detected_coords)
    assert smooth.ndim == 2 and smooth.shape[1] == 2
    assert smooth.shape[0] == 1000

    fitted = ja._fit_line_postprocessing(detected_coords)
    # returns fitted coords across same number of x points
    assert fitted.shape == detected_coords.shape


def test_map_detected_to_image_coords_and_compare():
    ja = JunctionAnalyzer(pixel_size_m=1e-6)
    W = 30
    H = 11

    # manual_line_rs: straight horizontal line across x with constant y
    manual_line_rs = np.column_stack([np.arange(W).astype(float), np.full(W, 50.0)])

    # detected_roi_coords: set row index equal to half so offset=0
    half = (H - 1) / 2.0
    detected_roi_coords = np.column_stack([np.arange(W).astype(float), np.full(W, half)])

    image_coords = ja._map_detected_to_image_coords(manual_line_rs, detected_roi_coords, roi_height=H)
    # With zero offset, image_coords should equal manual_line_rs (within float tolerance)
    assert np.allclose(image_coords, manual_line_rs, atol=1e-6)

    # compare with manual should give near-zero deviations
    mean_dev, std_dev, max_dev = ja._compare_with_manual(manual_line_rs, image_coords)
    assert pytest.approx(mean_dev, abs=1e-9) == 0.0
    assert pytest.approx(std_dev, abs=1e-9) == 0.0
    assert pytest.approx(max_dev, abs=1e-9) == 0.0


def test_detect_wrapper_runs_and_returns_results():
    H = 20
    W = 30
    roi = make_edge_roi(height=H, width=W, edge_row=10)
    # manual_line shorter than width to exercise resampling
    manual_line = np.column_stack([np.linspace(0, W - 1, 10), np.full(10, 50.0)])

    ja = JunctionAnalyzer(pixel_size_m=1e-6)
    results = ja.detect(roi, manual_line)

    # Should return a list (possibly empty if detection failed)
    assert isinstance(results, list)
    if results:
        name, detected_image_coords, metrics = results[0]
        assert isinstance(name, str)
        assert detected_image_coords.shape[1] == 2
        # metrics should be a tuple of three numeric values
        assert hasattr(metrics, '__len__') and len(metrics) == 3

