# roi_extractor.py
import numpy as np

def extract_line_rectangle(image, line_points, half_width_px):
    """
    Extract a rectangle along a line for junction detection.
    Returns ROI array and coordinates.
    """
    roi = np.zeros((2*half_width_px+1, len(line_points)))
    roi_coords = []

    for i, (x, y) in enumerate(line_points):
        if i == 0:
            dx = line_points[i+1,0] - x
            dy = line_points[i+1,1] - y
        else:
            dx = x - line_points[i-1,0]
            dy = y - line_points[i-1,1]

        perp_vec = np.array([-dy, dx])
        perp_vec = perp_vec / np.linalg.norm(perp_vec)
        col = []
        col_coords = []
        for offset in np.linspace(-half_width_px, half_width_px, 2*half_width_px+1):
            px = int(round(x + offset*perp_vec[0]))
            py = int(round(y + offset*perp_vec[1]))
            col.append(image[py, px] if 0<=px<image.shape[1] and 0<=py<image.shape[0] else 0.0)
            col_coords.append((px, py))
        roi[:,i] = col
        roi_coords.append(col_coords)

    roi_coords = np.array(roi_coords)  # shape: (line_length, 2*half_width+1, 2)
    return roi, roi_coords
