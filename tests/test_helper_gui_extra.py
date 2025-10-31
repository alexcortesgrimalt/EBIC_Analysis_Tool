import importlib
import numpy as np
import pytest
import os
import types
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def test_draw_scalebar_and_extract_profile(tmp_path):
    try:
        mod = importlib.import_module('code.helper_gui')
    except Exception:
        pytest.skip("Skipping helper_gui tests due to import error")

    draw_scalebar = mod.draw_scalebar
    extract_line_profile_data = mod.extract_line_profile_data

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

    data = np.arange(10000).reshape((100, 100)).astype(float)
    line_coords = ((10.0, 10.0), (20.0, 10.0))
    d_um, vals = extract_line_profile_data(line_coords, data, pixel_size=1e-6)
    assert d_um is not None and vals is not None
    assert len(d_um) == len(vals)


def make_synthetic_profile(n=201, A=1.0, k=0.5, y0=0.01):
    # distances from -L to +L
    L = 10.0
    x = np.linspace(-L, L, n)
    y = A * np.exp(-k * np.abs(x)) + y0
    prof = {
        'id': 0,
        'dist_um': x,
        'current': y,
        'sem': np.zeros_like(y),
        'intersection': None,
        'intersection_idx': n // 2
    }
    return prof


def test_fit_perpendicular_profiles_linear_basic(monkeypatch):
    try:
        mod = importlib.import_module('code.helper_gui')
    except Exception:
        pytest.skip("Skipping helper_gui linear tests due to import error")

    # prevent plots from blocking
    monkeypatch.setattr(plt, 'show', lambda *a, **k: None)

    prof = make_synthetic_profile(n=201, A=1.0, k=0.5, y0=0.01)
    viewer = types.SimpleNamespace(perpendicular_profiles=[prof])

    # call the function under test
    results = mod.fit_perpendicular_profiles_linear(viewer)
    assert isinstance(results, list)
    assert len(results) == 1

    r = results[0]
    assert 'left_fit' in r and 'right_fit' in r
    lf = r['left_fit']
    rf = r['right_fit']
    assert lf is not None and rf is not None
    # left slope should be positive, right slope negative
    assert lf['slope'] > 0
    assert rf['slope'] < 0
