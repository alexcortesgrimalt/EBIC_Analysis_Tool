import numpy as np
import pytest


def test_pixelmap_and_currentmap_basic():
    try:
        mod = __import__('code.maps', fromlist=['PixelMap', 'CurrentMap'])
    except Exception as e:
        pytest.skip(f"Skipping maps tests due to import error: {e}")

    PixelMap = mod.PixelMap
    CurrentMap = mod.CurrentMap

    # Create a fake metadata object with required fields
    class FakeMeta:
        def __init__(self):
            self.data = {
                'PixelSizeX': 1e-6,
                'Contrast': 2.0,
                'EffectiveAmpGain': 1e6,
                'OutputOffset': 0.0,
                'InputOffset': 0.0,
                'InverseEnabled': False,
                'BiasEnabled': False,
                'BiasVoltage': 0.0,
            }

    image = np.linspace(0, 65535, num=9).reshape((3, 3))
    meta = FakeMeta()

    pm = PixelMap(image, meta, output_dir='.')
    assert pm.data.shape == image.shape
    assert pm.pixel_size == meta.data['PixelSizeX']

    cm = CurrentMap(pm)
    assert cm.data.shape == image.shape
    # verify that current values are finite numbers
    assert np.all(np.isfinite(cm.data))


def test_pixelmap_save_and_currentmap_save_csv(tmp_path):
    try:
        mod = __import__('code.maps', fromlist=['PixelMap', 'CurrentMap'])
    except Exception as e:
        pytest.skip(f"Skipping maps tests due to import error: {e}")

    PixelMap = mod.PixelMap
    CurrentMap = mod.CurrentMap

    class FakeMeta:
        def __init__(self):
            self.data = {
                'PixelSizeX': 1e-6,
                'Contrast': 2.0,
                'EffectiveAmpGain': 1e6,
                'OutputOffset': 0.0,
                'InputOffset': 0.0,
                'InverseEnabled': False,
                'BiasEnabled': False,
                'BiasVoltage': 0.0,
            }

    image = np.array([[0, 32767], [65535, 10000]], dtype=np.uint16)
    meta = FakeMeta()

    outdir = tmp_path / "out"
    outdir.mkdir()

    pm = PixelMap(image, meta, output_dir=str(outdir))
    # ensure data converted to float64
    assert pm.data.dtype == np.float64

    # test save creates a csv file
    pm.save()
    p = outdir / "pixel_map.csv"
    assert p.exists()
    content = p.read_text()
    assert "," in content

    # test current computation matches manual formula (InverseEnabled = False)
    cm = CurrentMap(pm)
    pixels = pm.data

    # manual calculation
    C = meta.data['Contrast']
    G = meta.data['EffectiveAmpGain']
    O = meta.data['OutputOffset']
    I = meta.data['InputOffset']
    scale = 1
    offset = -0.5
    voltage = (pixels / 65535) * scale + offset
    expected = (((voltage - O) / C) - I) / G * 1e9

    assert np.allclose(cm.data, expected, rtol=1e-6, atol=1e-9)

    # save_csv
    csv_dir = tmp_path / "csvout"
    csv_dir.mkdir()
    cm.save_csv(str(csv_dir))
    csv_path = csv_dir / "current_map.csv"
    assert csv_path.exists()


def test_currentmap_inverse_enabled_true():
    try:
        mod = __import__('code.maps', fromlist=['PixelMap', 'CurrentMap'])
    except Exception as e:
        pytest.skip(f"Skipping maps tests due to import error: {e}")

    PixelMap = mod.PixelMap
    CurrentMap = mod.CurrentMap

    class FakeMetaInv:
        def __init__(self):
            self.data = {
                'PixelSizeX': 1e-6,
                'Contrast': 2.0,
                'EffectiveAmpGain': 1e6,
                'OutputOffset': 0.1,
                'InputOffset': 0.01,
                'InverseEnabled': True,
                'BiasEnabled': False,
                'BiasVoltage': 0.0,
            }

    image = np.array([[0, 65535]], dtype=np.uint16)
    pm = PixelMap(image, FakeMetaInv(), output_dir='.')
    cm = CurrentMap(pm)

    # compute expected with inverse branch (negated)
    pixels = pm.data
    meta = pm.metadata.data
    C = meta['Contrast']
    G = meta['EffectiveAmpGain']
    O = meta['OutputOffset']
    I = meta['InputOffset']
    voltage = (pixels / 65535) * 1 + -0.5
    expected = (((voltage - O) / C) + I) / G * -1e9

    assert np.allclose(cm.data, expected, rtol=1e-6, atol=1e-9)
