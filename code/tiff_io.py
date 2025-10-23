import os
import re
import glob
from .maps import PixelMap, CurrentMap
from .Metadata import Metadata
import numpy as np

class SEMTiffProcessor:
    def __init__(self):
        self.tiff_path = ""
        self.output_dir = ""
        self.metadata = None

    def _extract_xmp(self):
        with open(self.tiff_path, "rb") as f:
            content = f.read()
        pattern = re.compile(rb'<\?xpacket begin=.*?\?>.*?<\?xpacket end=.*?\?>', re.DOTALL)
        match = pattern.search(content)
        if not match:
            raise ValueError("No XML metadata found.")
        xmp_bytes = match.group(0)
        xml_str = xmp_bytes.decode("utf-8", errors="ignore").strip()
        xml_path = self.tiff_path.replace('.tif', '_metadata.xml')
        with open(xml_path, "w", encoding="utf-8") as out_file:
            out_file.write('<?xml version="1.0"?>\n' + xml_str)
        return xml_path

    def load_maps(self, output_root, sample_name):
        # Sort frame directories by their numeric suffix (frame_1, frame_2, ...)
        frame_dirs_unsorted = glob.glob(os.path.join(output_root, sample_name, "frame_*"))
        def frame_key(p):
            base = os.path.basename(p)
            import re
            m = re.search(r'frame_(\d+)', base)
            return int(m.group(1)) if m else 999999
        frame_dirs = sorted(frame_dirs_unsorted, key=frame_key)
        pixel_maps = []
        current_maps = []
        frame_sizes = []  # store (width, height) for each frame
        pixel_size = None

        for frame_dir in frame_dirs:
            pixel_path = os.path.join(frame_dir, "pixel_map.csv")
            current_path = os.path.join(frame_dir, "current_map.csv")

            if not os.path.exists(pixel_path):
                print(f"Warning: pixel_map.csv not found in {frame_dir}")
                pixel_data = None
                width, height = 0, 0
            else:
                pixel_data = np.loadtxt(pixel_path, delimiter=",")
                height, width = pixel_data.shape

            if not os.path.exists(current_path):
                current_data = None
            else:
                current_data = np.loadtxt(current_path, delimiter=",")

            pixel_maps.append(pixel_data)
            current_maps.append(current_data)
            frame_sizes.append((width, height))  # store frame dimensions

            if pixel_size is None and pixel_data is not None:
                pixel_size = self.metadata.data['PixelSizeX']

        return pixel_maps, current_maps, pixel_size, sample_name,frame_sizes


    def _process_tiff_file(self, tiff_file, output_dir, show=False):
        """General method to process a TIFF file and save results in output_dir.

        This restores the original processing flow: use PIL Image.open and
        iterate with seek(frame_index), saving each frame's pixel_map and
        for frames > 0 compute a CurrentMap(pixel_map) and save_csv —
        allow exceptions to propagate so caller sees errors.
        """
        import os
        from PIL import Image

        self.tiff_path = tiff_file
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Extract metadata
        self.metadata = Metadata(self._extract_xmp())

        # Open TIFF with PIL and iterate frames using seek
        img = Image.open(self.tiff_path)
        frame_index = 0

        while True:
            try:
                img.seek(frame_index)
                frame = img.copy()
                frame_dir = os.path.join(self.output_dir, f"frame_{frame_index + 1}")
                os.makedirs(frame_dir, exist_ok=True)

                # Create and save pixel map
                pixel_map = PixelMap(frame, self.metadata, frame_dir)
                pixel_map.save()

                # Create and save current map (skip first frame)
                if frame_index > 0:
                    current_map = CurrentMap(pixel_map)
                    current_map.save_csv(frame_dir)

                frame_index += 1
            except EOFError:
                break

        if show:
            print(f"Finished processing {tiff_file} -> {self.output_dir}")

    def process_single(self, tiff_file, output_root="tiff_test_output", show=False):
        """Process a single TIFF file."""
        file_stem = os.path.splitext(os.path.basename(tiff_file))[0]
        output_dir = os.path.join(output_root, file_stem)
        self._process_tiff_file(tiff_file, output_dir, show=show)

    def process_tiff(self, tiff_path, output_root, sample_name):
        """
        Backwards-compatible method used by some callers (PixelMatching):
        - Reads the TIFF and expects at least two frames (frame_1 pixel map, frame_2 current map)
        - Saves frames into output_root/sample_name/frame_N
        - Returns the same tuple as load_maps(output_root, sample_name)
        """
        # ensure output directory exists
        out_dir = os.path.join(output_root, sample_name)
        os.makedirs(out_dir, exist_ok=True)

        # Use our existing _process_tiff_file which already writes frames and current maps
        self._process_tiff_file(tiff_path, out_dir, show=False)

        # After writing, load and return maps
        return self.load_maps(output_root, sample_name)

    def process_multiple(self, folder_path, output_root="output", show=False):
        """Process all TIFF files in a folder."""
        import glob, os

        tiff_paths = sorted(glob.glob(os.path.join(folder_path, "*.tif")))
        for tiff_path in tiff_paths:
            file_stem = os.path.splitext(os.path.basename(tiff_path))[0]
            output_dir = os.path.join(output_root, file_stem)
            self._process_tiff_file(tiff_path, output_dir, show=show)

