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

        def get_nested_value(tag, default=0.0):
            """Try to read <tag><rdf:value>numeric</rdf:value></tag>, fallback to direct text.

            Returns float(default) if value not found or not convertable.
            """
            elem = desc.find(tag, ns)
            if elem is not None:
                # prefer nested rdf:value
                val = elem.find('rdf:value', ns)
                if val is not None and val.text is not None:
                    try:
                        return float(val.text.strip())
                    except Exception:
                        pass
                # fallback to direct text inside the tag
                if elem.text is not None:
                    try:
                        return float(elem.text.strip())
                    except Exception:
                        pass
            try:
                return float(default)
            except Exception:
                return 0.0

        return {
            'PixelSizeX': float(desc.findtext('image:PixelSizeX', '1e-6', namespaces=ns)),
            'Contrast': get_nested_value('efa:Contrast', 0.0),
            'EffectiveAmpGain': get_nested_value('efa:EffectiveAmpGain', 1e6),
            'OutputOffset': get_nested_value('efa:OutputOffset', 0.0),
            'InputOffset': get_nested_value('efa:InputOffset', 0.0),
            'InverseEnabled': bool(int(desc.findtext('efa:InverseEnabled', '0', namespaces=ns))),
            'BiasEnabled': bool(int(desc.findtext('efa:BiasEnabled', '0', namespaces=ns))),
            'BiasVoltage': get_nested_value('efa:Bias', 0.0)
        }

