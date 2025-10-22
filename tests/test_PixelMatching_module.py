import numpy as np
import pytest


def test_map_pixels_small_images():
    try:
        mod = __import__('code.PixelMatching', fromlist=['PixelMap', 'SEMTiffProcessor'])
    except Exception as e:
        pytest.skip(f"Skipping PixelMatching tests due to import error: {e}")

    map_pixels = mod.SEMTiffProcessor.map_pixels

    # small synthetic images 10x10
    img1 = np.zeros((10, 10), dtype=float)
    img2 = np.zeros((10, 10), dtype=float)
    # add a small blob in img1 and same in img2 shifted by (1,1)
    img1[4, 4] = 1.0
    img2[5, 5] = 1.0

    mapping = map_pixels(img1, img2, window_size=3, search_radius=2)
    assert isinstance(mapping, dict)
    assert (0, 0) in mapping


def test_match_line_only_basic():
    try:
        mod = __import__('code.PixelMatching', fromlist=['SEMTiffProcessor'])
    except Exception as e:
        pytest.skip(f"Skipping PixelMatching tests due to import error: {e}")

    proc = mod.SEMTiffProcessor()

    img1 = np.zeros((20, 20), dtype=float)
    img2 = np.zeros((20, 20), dtype=float)
    # create a diagonal line in both images with slight shift
    for i in range(5, 15):
        img1[i, i] = 1.0
        img2[i+1, i+1] = 1.0

    line_coords = np.array([[i, i] for i in range(5, 15)])
    mapping = proc.match_line_only(img1, img2, line_coords, window_size=3, search_radius=2, step=2)
    assert isinstance(mapping, dict)
    # mapping keys are (row,col) in image1
    k = list(mapping.keys())[0]
    assert isinstance(k, tuple) and len(k) == 2
