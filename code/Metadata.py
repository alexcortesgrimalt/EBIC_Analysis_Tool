import xml.etree.ElementTree as ET

class Metadata:
    def __init__(self, xml_path):
        self.xml_path = xml_path
        self.data = self._parse_metadata()

    def _parse_metadata(self):
        ns = {
            'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
            'image': 'http://ns.pointelectronic.com/Image/1.0/',
            'efa': 'http://ns.pointelectronic.com/EFA/1.0/',
        }

        tree = ET.parse(self.xml_path)
        root = tree.getroot()
        desc = root.find('.//rdf:Description', ns)

        def get_nested_value(tag):
            elem = desc.find(tag, ns)
            if elem is not None:
                val = elem.find('rdf:value', ns)
                if val is not None:
                    return float(val.text)
            return 0.0

        return {
            'PixelSizeX': float(desc.findtext('image:PixelSizeX', '1e-6', namespaces=ns)),
            'Contrast': get_nested_value('efa:Contrast'),
            'EffectiveAmpGain': float(desc.findtext('efa:EffectiveAmpGain', '1e6', namespaces=ns)),
            'OutputOffset': get_nested_value('efa:OutputOffset'),
            'InputOffset': get_nested_value('efa:InputOffset'),
            'InverseEnabled': bool(int(desc.findtext('efa:InverseEnabled', '0', namespaces=ns))),
            'BiasEnabled': bool(int(desc.findtext('efa:BiasEnabled', '0', namespaces=ns))),
            'BiasVoltage': get_nested_value('efa:Bias')
        }

