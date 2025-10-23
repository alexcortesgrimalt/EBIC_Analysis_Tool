"""Launcher that runs the project package entrypoint.

Use this when you want to run the application from the repo root as a script.
It imports the `code` package and executes `code.main` as a module so package-relative
imports work correctly.
"""
import importlib
import sys
from pathlib import Path

if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    module = importlib.import_module("code.main")

    if hasattr(module, 'main') and callable(module.main):
        module.main()
