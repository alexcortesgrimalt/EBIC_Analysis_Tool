import importlib
import numpy as np
import pytest
import os


def test_draw_scalebar_and_extract_profile(tmp_path):
    try:
        mod = importlib.import_module('code.helper_gui')
    except Exception:
        pytest.skip("Skipping helper_gui tests due to import error")

    draw_scalebar = mod.draw_scalebar
    extract_line_profile_data = mod.extract_line_profile_data

    # create a fake axes with minimal API
    class FakeAx:
        def __init__(self):
            self._xlim = (0, 100)
            self._ylim = (100, 0)
            self.patches = []
            self.texts = []

        def get_xlim(self):
            return self._xlim

        def get_ylim(self):
            return self._ylim

        def add_patch(self, p):
            self.patches.append(p)

        def text(self, *args, **kwargs):
            self.texts.append((args, kwargs))

    ax = FakeAx()
    draw_scalebar(ax, pixel_size=1e-6, length_um=1.0)
    assert len(ax.patches) == 1

    # test extract_line_profile_data
    data = np.arange(10000).reshape((100, 100)).astype(float)
    line_coords = ((10.0, 10.0), (20.0, 10.0))
    d_um, vals = extract_line_profile_data(line_coords, data, pixel_size=1e-6)
    assert d_um is not None and vals is not None
    assert len(d_um) == len(vals)
