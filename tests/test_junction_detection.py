import numpy as np
import os


def test_junction_detection_on_synthetic_edge(tmp_path):
    """Create a synthetic ROI with a clear tilted interface and assert detection is close."""
    # Import here so tests don't fail if package import path isn't set until pytest run
    from code.Junction_Analyser import JunctionAnalyzer

    H = 60
    W = 120

    # Create a tilted edge: edge_row = base + slope * col
    base = 22
    slope = 0.06
    edge_rows = (base + slope * np.arange(W)).astype(int)

    # Build ROI: above edge -> low intensity, below edge -> high intensity
    roi = np.zeros((H, W), dtype=np.uint8)
    for col in range(W):
        r = edge_rows[col]
        roi[:r, col] = 40
        roi[r:, col] = 200

    # Add small Gaussian noise
    rng = np.random.default_rng(0)
    roi = np.clip(roi + (rng.normal(scale=2.0, size=roi.shape)), 0, 255).astype(np.uint8)

    # Manual line: straight center line across image (will be resampled inside detect)
    manual_line = np.column_stack([np.linspace(0, W - 1, W), np.full(W, H / 2)])

    ja = JunctionAnalyzer(pixel_size_m=1e-6)
    results = ja.detect(roi, manual_line)

    assert results, "No detection results returned"

    # Get detected coords from first method
    name, detected_coords, metrics = results[0]

    # Compare detected y positions to ground-truth edge rows
    detected_rows = detected_coords[:, 1]

    # Because detected_coords are floats and mapping may introduce small offsets,
    # assert mean absolute error is small (e.g., < 2 pixels) and max error reasonable.
    mae = np.mean(np.abs(detected_rows - edge_rows))
    max_err = np.max(np.abs(detected_rows - edge_rows))

    assert mae < 2.0, f"Mean absolute error too large: {mae} px"
    assert max_err < 6.0, f"Max error too large: {max_err} px"
