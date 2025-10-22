import numpy as np
from code.ROI_extractor import extract_line_rectangle


def test_extract_line_rectangle_shape_and_coords():
    img = np.zeros((20, 20), dtype=float)
    # line from (5,10) to (15,10)
    line = np.column_stack([np.linspace(5,15,11), np.linspace(10,10,11)])
    roi, coords = extract_line_rectangle(img, line, half_width_px=2)
    assert roi.shape[0] == 2*2 + 1
    assert roi.shape[1] == len(line)
    assert coords.shape[0] == len(line)
    assert coords.shape[1] == 2*2 + 1
