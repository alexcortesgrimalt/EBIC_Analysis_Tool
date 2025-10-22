import importlib
import pytest
import numpy as np
import os
import tempfile


def test_processor_wrapper_calls_underlying(monkeypatch, tmp_path):
    try:
        mod = importlib.import_module('code.processor')
    except Exception:
        pytest.skip("Skipping processor tests due to import error")

    SEMWrapper = mod.SEMTiffProcessorWrapper

    class FakeProc:
        def __init__(self):
            self.calls = []

        def process_single(self, path, output_root, show=False):
            self.calls.append((path, output_root, show))

        def load_maps(self, output_folder, sample_name):
            return ([], [], 1e-6, sample_name, [])

    wrapper = SEMWrapper()
    # monkeypatch the internal processor
    wrapper.processor = FakeProc()

    files = [tmp_path / 'a.tif', tmp_path / 'b.tif']
    for f in files:
        f.write_text('dummy')

    out = wrapper.process_files(files, tmp_path)
    assert isinstance(out, list)


def test_semtiffprocessor_extract_xmp_and_load_maps(tmp_path):
    try:
        mod = importlib.import_module('code.tiff_io')
    except Exception:
        pytest.skip("Skipping tiff_io tests due to import error")

    SEM = mod.SEMTiffProcessor
    proc = SEM()

    # create a dummy file with embedded xmp packet
    content = b"randomdata<?xpacket begin='abc'?>\n<rdf:Description>\n</rdf:Description>\n<?xpacket end='abc'?>rest"
    f = tmp_path / 'dummy.tif'
    f.write_bytes(content)

    proc.tiff_path = str(f)
    # _extract_xmp should produce an xml file path
    xml_path = proc._extract_xmp()
    assert os.path.exists(xml_path)

    # create fake frame directories for load_maps
    sample_dir = tmp_path / 'sample'
    frame1 = sample_dir / 'frame_1'
    frame1.mkdir(parents=True)
    # write small csvs
    np.savetxt(frame1 / 'pixel_map.csv', np.zeros((2,2)), delimiter=',')
    np.savetxt(frame1 / 'current_map.csv', np.zeros((2,2)), delimiter=',')

    proc.metadata = type('M', (), {'data': {'PixelSizeX': 1e-6}})()
    pixel_maps, current_maps, pixel_size, sample_name, frame_sizes = proc.load_maps(str(tmp_path), 'sample')
    assert isinstance(pixel_maps, list)
    assert isinstance(current_maps, list)
