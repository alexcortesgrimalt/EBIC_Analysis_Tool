This folder contains pytest-based tests for the `code/` package.

Quick PowerShell commands to run the tests (use your activated environment):

```powershell
# go to project root
Set-Location -Path 'C:\Users\biselli\Desktop\Code\MasterThesis\EBIC_Analysis_Tool'

# ensure pytest is installed (no-op if already present)
python -m pip install --upgrade pip setuptools wheel ; python -m pip install pytest

# run full test suite (quiet)
python -m pytest -q

# run with verbose reporting and show skip/fail reasons
python -m pytest -q -rA
```

Notes:
- Tests are designed to be lightweight and avoid executing functions which may
  have GUI or hardware dependencies. Modules that cannot be imported will be
  skipped with the import error shown in pytest output.
