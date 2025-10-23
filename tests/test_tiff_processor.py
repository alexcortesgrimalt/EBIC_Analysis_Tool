import os
import io
import sys
import pathlib
import numpy as np
from PIL import Image
import textwrap

# Ensure repository root is on sys.path so `code` package is importable during pytest runs
repo_root = pathlib.Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def _make_xmp_packet():
    # Minimal XMP packet matching Metadata._parse_metadata expectations
    xmp = textwrap.dedent('''
    <?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>
    <x:xmpmeta xmlns:x="adobe:ns:meta/">
      <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
        <rdf:Description xmlns:efa="http://ns.pointelectronic.com/EFA/1.0/" xmlns:image="http://ns.pointelectronic.com/Image/1.0/">
          <image:PixelSizeX>1e-6</image:PixelSizeX>
          <efa:Contrast><rdf:value>2.0</rdf:value></efa:Contrast>
          <efa:EffectiveAmpGain>1000000</efa:EffectiveAmpGain>
          <efa:OutputOffset><rdf:value>0.0</rdf:value></efa:OutputOffset>
          <efa:InputOffset><rdf:value>0.0</rdf:value></efa:InputOffset>
          <efa:InverseEnabled>0</efa:InverseEnabled>
          <efa:BiasEnabled>0</efa:BiasEnabled>
          <efa:Bias><rdf:value>0.0</rdf:value></efa:Bias>
        </rdf:Description>
      </rdf:RDF>
    </x:xmpmeta>
    <?xpacket end="w"?>
    ''')
    return xmp.encode('utf-8')


def _write_2frame_tiff_with_xmp(path):
    # Create two simple 16-bit frames
    a = (np.linspace(0, 65535, num=64*64, dtype=np.uint16).reshape((64, 64)))
    b = np.roll(a, 100).astype(np.uint16)
    img1 = Image.fromarray(a)
    img2 = Image.fromarray(b)

    # Save multi-page TIFF
    img1.save(path, format='TIFF', save_all=True, append_images=[img2])

    # Append an XMP packet to the file so SEMTiffProcessor._extract_xmp finds it
    with open(path, 'ab') as fh:
        fh.write(b"\n")
        fh.write(_make_xmp_packet())


def test_semtiffprocessor_writes_second_frame_current_map(tmp_path):
    """Regression test: ensure a 2-frame TIFF with XMP produces current_map for frame 2."""
    # Ensure our repository 'code' package is imported (avoid stdlib 'code')
    if 'code' in sys.modules:
        del sys.modules['code']
    from importlib import import_module
    tiff_mod = import_module('code.tiff_io')
    SEMTiffProcessor = tiff_mod.SEMTiffProcessor

    tiff_path = tmp_path / "two_frames.tif"
    out_root = tmp_path / "out"
    sample_name = "sample_test"

    _write_2frame_tiff_with_xmp(str(tiff_path))

    proc = SEMTiffProcessor()

    # Use process_tiff which returns (pixel_maps, current_maps, pixel_size, sample_name, frame_sizes)
    pixel_maps, current_maps, pixel_size, sn, frame_sizes = proc.process_tiff(str(tiff_path), str(out_root), sample_name)

    # We expect at least two frames
    assert len(pixel_maps) >= 2, "pixel_maps must contain at least two frames"
    assert len(current_maps) >= 2, "current_maps must contain at least two entries"

    # First frame: pixel map exists
    assert pixel_maps[0] is not None
    # Second frame: pixel map exists
    assert pixel_maps[1] is not None

    # Second frame: current map should be produced and numeric
    assert current_maps[1] is not None, "current_map for second frame should be present"
    arr = np.array(current_maps[1])
    assert arr.size > 0
    assert np.isfinite(arr).all(), "current_map must contain finite numeric values"

    # Also check CSV file was written on disk
    csv_path = os.path.join(str(out_root), sample_name, 'frame_2', 'current_map.csv')
    assert os.path.exists(csv_path), f"expected current_map.csv at {csv_path}"
