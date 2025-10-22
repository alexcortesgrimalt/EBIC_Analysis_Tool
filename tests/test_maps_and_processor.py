import tempfile
import os
import numpy as np
from code.maps import PixelMap, CurrentMap
from code.processor import SEMTiffProcessorWrapper


def test_pixel_and_current_map_basic(tmp_path):
    # create fake metadata object
    class FakeMeta:
        def __init__(self):
            self.data = {'PixelSizeX': 1e-6, 'Contrast': 1.0, 'EffectiveAmpGain': 1e6,
                         'OutputOffset': 0.0, 'InputOffset': 0.0, 'InverseEnabled': 0, 'BiasEnabled': 0, 'BiasVoltage': 0.0}

    meta = FakeMeta()
    img = np.zeros((10, 10), dtype=float)
    outdir = tmp_path / "out"
    outdir.mkdir()

    pm = PixelMap(img, meta, str(outdir))
    pm.save()
    csv_path = outdir / "pixel_map.csv"
    assert csv_path.exists()

    cm = CurrentMap(pm)
    assert cm.data.shape == img.shape


def test_processor_wrapper_importable():
    wrapper = SEMTiffProcessorWrapper()
    # wrapper.processor should have methods like process_single or process_tiff
    assert hasattr(wrapper.processor, 'load_maps')
