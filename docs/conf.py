# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from pathlib import Path

import importlib
from typing import Any

try:
    tomllib: Any = importlib.import_module("tomllib")  # type: ignore
except ModuleNotFoundError:
    # Fallback for older Python versions: tomli provides compatible API
    tomllib = importlib.import_module("tomli")  # type: ignore

# -- Path setup --------------------------------------------------------------
# Ensure the project package is importable for autodoc (assumes src layout)
sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "CTG-wave"
copyright = "2025, Andrea Scaglioni"
author = "Andrea Scaglioni"
release = "0.1.0"

# Read project version from pyproject.toml
pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
release = "0.0.0"
if pyproject_path.exists():
    with pyproject_path.open("rb") as f:
        data = tomllib.load(f)
    # PEP 621
    release = (
        data.get("project", {}).get("version")
        or data.get("tool", {}).get("poetry", {}).get("version")
        or release
    )


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinx_mdinclude",
    "sphinx_copybutton",
    "sphinxcontrib.spelling",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

# Use the docs static logo now stored in docs/_static
# html_logo = "_static/logo.svg"

# # Set a concise HTML title (avoid theme/appending 'Documentation')
html_title = "CTG-wave 0.1.0"
# # Optional short title used in smaller viewports or sidebars
html_short_title = "CTG-wave 0.1.0"

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_typehints = "description"

# Autosummary: generate pages for modules listed with ``.. autosummary::``. This makes the API section (docs/api.rst) expand into individual rst files under docs/generated/ when building the docs.
autosummary_generate = True
# Do not include imported members in autosummary listings by default.
# When True, autosummary will include members that were imported into a
# module (for example ``from math import ceil``); that often produces
# extraneous entries in the API pages. Prefer explicit ``__all__`` or
# per-module autosummary entries when you want imported symbols listed.
autosummary_imported_members = False
# Mock dependencies (package that are imported at module import time but not available in the docs build env) so that Sphinx can import project modules
autodoc_mock_imports = [
    "pydantic",
    "dolfinx",
    "mpi4py",
    "petsc4py",
]
# Optional: link to other projects
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

# Helpful external references used across the docs. These allow Sphinx to
# cross-reference symbols from common scientific packages used in the
# project (NumPy, SciPy, Matplotlib, Pydantic, MPI for Python).
intersphinx_mapping.update(
    {
        "numpy": ("https://numpy.org/doc/stable/", None),
        "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
        "matplotlib": ("https://matplotlib.org/stable/", None),
        "pydantic": ("https://docs.pydantic.dev/latest/", None),
        "mpi4py": ("https://mpi4py.readthedocs.io/en/stable/", None),
    }
)

# Spelling check options
spelling_lang = "en_US"
spelling_word_list_filename = "spelling_wordlist.txt"
spelling_show_suggestions = True
