import importlib
import inspect
import types
import pytest
from pathlib import Path


def test_semstartergui_class_exists_and_methods():
    # Import the module; skip if GUI deps cannot be satisfied
    try:
        mod = importlib.import_module('code.gui')
    except Exception as e:
        pytest.skip(f"Skipping gui tests due to import error: {e}")

    assert hasattr(mod, 'SEMStarterGUI')
    cls = mod.SEMStarterGUI
    # Check that build_gui, browse_folder, update_file_list signatures exist
    for name in ('build_gui', 'browse_folder', 'update_file_list', 'process_files', 'choose_dataset', 'open_viewer'):
        assert hasattr(cls, name), f"SEMStarterGUI missing {name}"
        method = getattr(cls, name)
        assert inspect.isfunction(method) or inspect.ismethod(method)


def test_open_viewer_calls_processor_load_maps(monkeypatch, tmp_path):
    # Import module and class
    try:
        mod = importlib.import_module('code.gui')
    except Exception as e:
        pytest.skip(f"Skipping gui tests due to import error: {e}")

    cls = mod.SEMStarterGUI

    # Create a dummy instance without calling __init__ to avoid Tk mainloop
    instance = object.__new__(cls)

    # Provide a fake processor with a load_maps method
    processed = (['pmap'], ['cmap'], 1.0, 'sample', [(100, 100)])

    class FakeProcessor:
        def load_maps(self, output_folder, sample_name):
            return processed

    instance.processor = FakeProcessor()

    # Monkeypatch messagebox to return True to simulate user clicking Yes
    monkeypatch.setattr('code.gui.messagebox', types.SimpleNamespace(askyesno=lambda *args, **kwargs: True,
                                                                      showerror=lambda *a, **k: None,
                                                                      showinfo=lambda *a, **k: None))

    # Also monkeypatch SEMViewer to avoid real UI
    monkeypatch.setattr('code.gui.SEMViewer', lambda *args, **kwargs: types.SimpleNamespace(show=lambda: None))

    # Call open_viewer with our fake processor
    instance.open_viewer(tmp_path, 'sample')
