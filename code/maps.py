
import numpy as np
import pandas as pd
import os
from Metadata import Metadata

class PixelMap:
    def __init__(self, image, metadata: Metadata, output_dir):
        self.data = np.array(image).astype(np.float64)
        self.metadata = metadata
        self.output_dir = output_dir
        self.pixel_size = metadata.data['PixelSizeX']

    def save(self):
        path = os.path.join(self.output_dir, "pixel_map.csv")
        pd.DataFrame(self.data).to_csv(path, header=False, index=False)


class CurrentMap:
    def __init__(self, pixel_map: PixelMap):
        self.metadata = pixel_map.metadata.data
        self.pixel_size = pixel_map.pixel_size
        self.data = self._compute_current(pixel_map.data)

    def _compute_current(self, pixels):
        C = self.metadata['Contrast']
        G = self.metadata['EffectiveAmpGain']
        O = self.metadata['OutputOffset']
        I = self.metadata['InputOffset']
        inv = self.metadata['InverseEnabled']
        bias_enabled = self.metadata['BiasEnabled']
        bias_voltage = self.metadata['BiasVoltage']

        scale = 1  # mV
        offset = -0.5  # mV
        voltage = (pixels / 65535) * scale + offset

        #if bias_enabled:
        #    voltage -= bias_voltage

        if inv:
            return (((voltage - O) / C) + I) / G * -1e9
        else:
            return (((voltage - O) / C) - I) / G * +1e9

    def save_csv(self, folder_path):
        path = os.path.join(folder_path, "current_map.csv")
        pd.DataFrame(self.data).to_csv(path, header=False, index=False)

