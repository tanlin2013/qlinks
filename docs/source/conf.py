"""Sphinx configuration for the qlinks documentation."""

from __future__ import annotations

import os  # noqa: F401
import sys
from importlib import metadata
from pathlib import Path

DOCS_SOURCE_DIR = Path(__file__).resolve().parent
DOCS_DIR = DOCS_SOURCE_DIR.parent
PROJECT_ROOT = DOCS_DIR.parent

sys.path.insert(0, str(PROJECT_ROOT))

# -- Project information -----------------------------------------------------

project = "qlinks"
copyright = "2023, Tan Tao-Lin"
author = "Tan Tao-Lin"

try:
    release = metadata.version("qlinks")
except metadata.PackageNotFoundError:
    release = "0+unknown"

version = release

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.githubpages",
    "sphinx.ext.ifconfig",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "nbsphinx",
    "m2r2",
]

templates_path = ["_templates"]
source_suffix = [".rst", ".md"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", ".ipynb"]

# Optional runtime extras are not required to build the API reference.  Mocking
# them keeps the documentation build usable with only the docs dependency group.
autodoc_mock_imports = [
    "cupy",
    "cupyx",
    "h5py",
    "igraph",
    "ortools",
    "plotly",
    "pynauty",
    "pyvis",
    "ray",
]

# Document class-level and __init__ docstrings together.  This matches the
# dataclass-heavy style used by the public API.
autoclass_content = "both"

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": False,
    "exclude-members": "__weakref__",
    "show-inheritance": True,
}
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"

autosummary_generate = True

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True

nbsphinx_allow_errors = True

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_title = "qlinks"

html_theme_options = {
    "github_url": "https://github.com/tanlin2013/qlinks",
    "repository_url": "https://github.com/tanlin2013/qlinks",
    "repository_branch": "main",
    "path_to_docs": "docs/source",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "use_fullscreen_button": False,
    "use_download_button": False,
    "home_page_in_toc": True,
}

# -- Link with documentation of external projects ----------------------------

intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable/", None),
    "python": ("https://docs.python.org/3", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
}

# -- Options for todo extension ----------------------------------------------

todo_include_todos = True
