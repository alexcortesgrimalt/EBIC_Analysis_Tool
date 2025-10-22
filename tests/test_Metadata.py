import os
import tempfile
from code.Metadata import Metadata


def test_metadata_parsing_minimal_xml():
    # Create a minimal XMP-like XML with required nodes
    xml_content = '''<?xml version="1.0"?>
    <x:xmpmeta xmlns:x="adobe:ns:meta/">
      <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
        <rdf:Description xmlns:image="http://ns.pointelectronic.com/Image/1.0/"
                         xmlns:efa="http://ns.pointelectronic.com/EFA/1.0/">
          <image:PixelSizeX>2e-6</image:PixelSizeX>
          <efa:EffectiveAmpGain>
            <rdf:value>2e6</rdf:value>
          </efa:EffectiveAmpGain>
          <efa:InverseEnabled>0</efa:InverseEnabled>
          <efa:BiasEnabled>1</efa:BiasEnabled>
        </rdf:Description>
      </rdf:RDF>
    </x:xmpmeta>'''

    fd, path = tempfile.mkstemp(suffix='.xml')
    os.close(fd)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(xml_content)

    meta = Metadata(path)
    assert isinstance(meta.data, dict)
    assert abs(meta.data['PixelSizeX'] - 2e-6) < 1e-12
    assert meta.data['InverseEnabled'] is False
    assert meta.data['BiasEnabled'] is True

    os.remove(path)
