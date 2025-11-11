from pathlib import Path
from .tiff_io import SEMTiffProcessor

class SEMTiffProcessorWrapper:
    def __init__(self):
        self.processor = SEMTiffProcessor()

    def process_files(self, files: list[Path], output_folder: Path) -> list[str]:
        processed_names = []

        for f in files:
            self.processor.process_single(str(f), output_root=str(output_folder), show=False)
            processed_names.append(f.stem)

        return processed_names

    def load_maps(self, output_folder: Path, sample_name: str):
        return self.processor.load_maps(str(output_folder), sample_name)

    def load_multiple_maps(self, output_folder: Path, sample_names: list[str]):
        """Load maps for multiple processed sample names.

        Returns a list of tuples as returned by load_maps.
        """
        results = []
        for name in sample_names:
            results.append(self.processor.load_maps(str(output_folder), name))
        return results
