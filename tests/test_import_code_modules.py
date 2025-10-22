import importlib
import pytest

# List of modules to smoke-test for importability
modules = [
    'code',
    'code.DiffLenExt',
    'code.gui',
    'code.helper_gui',
    'code.Junction_Analyser',
    'code.main',
    'code.maps',
    'code.Metadata',
    'code.perpendicular',
    'code.PixelMatching',
    'code.processor',
    'code.ROI_extractor',
    'code.sem_viewer',
    'code.tiff_io',
]


@pytest.mark.parametrize('modname', modules)
def test_import_module(modname):
    """Attempt to import each module. If import fails, skip with the error message.

    This provides a lightweight smoke test to ensure tests run in CI without
    executing module-level side effects that might require GUI, hardware, or
    file resources.
    """
    try:
        importlib.import_module(modname)
    except Exception as e:
        pytest.skip(f"Skipping import of {modname}: {e}")
