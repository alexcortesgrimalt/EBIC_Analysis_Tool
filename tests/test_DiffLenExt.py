import numpy as np
import inspect
import math
import pytest

from code.DiffLenExt import DiffusionLengthExtractor


def test_exp_models_basic():
    x = np.array([0.0, 1.0, 2.0])
    A, lam, y0, x0 = 2.0, 0.5, 1.0, 0.0

    # exp_model and exp_falling use same formula in this module
    expected = A * np.exp(-lam * (x - x0)) + y0

    assert np.allclose(DiffusionLengthExtractor.exp_model(x, A, lam, y0, x0), expected)
    assert np.allclose(DiffusionLengthExtractor.exp_falling(x, A, lam, y0, x0), expected)


def test_calculate_r2_perfect_and_constant():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    y_fit = y.copy()
    r2 = DiffusionLengthExtractor._calculate_r2(y, y_fit)
    assert pytest.approx(r2, rel=1e-9) == 1.0

    # constant signal -> ss_tot = 0 -> returns nan
    y_const = np.ones(5)
    y_fit_const = np.ones(5)
    r2_const = DiffusionLengthExtractor._calculate_r2(y_const, y_fit_const)
    assert math.isnan(r2_const)


def test_find_snr_end_index_behaviour():
    dle = DiffusionLengthExtractor()

    # Make a signal which decays and has a noisy tail with low-level values
    head = np.linspace(10.0, 2.0, 20)
    tail = np.ones(10) * 0.1 + np.random.default_rng(0).normal(scale=0.01, size=10)
    y = np.concatenate([head, tail])

    # Use a modest SNR threshold; should return an index larger than min_points
    idx = dle._find_snr_end_index(y, snr_threshold=2.0, min_points=5)
    assert isinstance(idx, int)
    assert idx >= 5
    assert idx <= len(y)


def test_apply_low_pass_filter_preserves_length_and_mean():
    dle = DiffusionLengthExtractor(pixel_size=1e-6)
    rng = np.random.default_rng(1)
    y = rng.normal(loc=0.5, scale=0.2, size=128)

    filtered = dle.apply_low_pass_filter(y, cutoff_fraction=0.2, visualize=False)
    assert filtered.shape == y.shape
    # mean should be approximately preserved
    assert abs(np.mean(filtered) - np.mean(y)) < 1e-6


def test_compute_average_lengths_quantization():
    # choose pixel_size so pixel_size_um = 1.0 µm for easy reasoning
    dle = DiffusionLengthExtractor(pixel_size=1e-6)

    # Create fake results: one profile with left=1.2, right=1.6, depletion=2.3
    dle.results = [
        {
            'Profile': 1,
            'depletion': {'depletion_width': 2.3},
            'fit_sides': [
                {'side': 'Left', 'inv_lambda': 1.2},
                {'side': 'Right', 'inv_lambda': 1.6},
            ]
        }
    ]

    out = dle.compute_average_lengths(show_table=False)
    # pixel_size_um = 1.0, so values should be rounded to nearest 1.0
    assert out['average_depletion_width'] == 2.0
    assert out['average_diffusion_length_left'] == 1.0
    assert out['average_diffusion_length_right'] == 2.0
    # combined average (1.2+1.6)/2 = 1.4 -> rounds to 1.0
    assert out['average_diffusion_length_all'] == 1.0
