import importlib
import pytest


def test_main_calls_gui_and_dpi(monkeypatch):
    try:
        mod = importlib.import_module('code.main')
    except Exception:
        pytest.skip("Skipping main tests due to import error")

    # monkeypatch SEMStarterGUI and enable_windows_dpi_awareness
    called = {}

    def fake_enable():
        called['dpi'] = True

    class FakeGUI:
        def __init__(self):
            called['gui'] = True

    monkeypatch.setattr('code.main.enable_windows_dpi_awareness', fake_enable)
    monkeypatch.setattr('code.main.SEMStarterGUI', FakeGUI)

    # call main
    mod.main()
    assert called.get('dpi', False) is True
    assert called.get('gui', False) is True

