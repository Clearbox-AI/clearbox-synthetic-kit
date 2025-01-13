# Configuration file for the Sphinx documentation builder.

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))  # Adjust the path as needed

project = 'clearbox-synthetic-kit'
copyright = '2025, Clearbox-AI'
author = 'Clearbox-AI'

extensions = [
    'sphinx.ext.autodoc',  # Automatically document code
    'sphinx.ext.napoleon',  # Support for Google-style and NumPy-style docstrings
    'sphinx.ext.viewcode',  # Add links to source code
    'sphinx.ext.intersphinx',  # Link to other project's documentation
    'myst_parser', # To include Markdown files - if not installed when running 'make html' run -> pip install myst-parser
    'sphinx_rtd_theme'
]

myst_enable_extensions = [
    "html_image", # Allows html images format conversion
    "linkify"
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}

templates_path = ['_templates']
exclude_patterns = ['diffusion/diffusion.py',
                    'engine/engine.py',
                    'metrics/privacy/gower_matrix_c.pyx',
                    'VAE/vae.py']

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "logo_only": True,
    "style_nav_header_background": "#483a8f",
}
html_logo = "img/cb_white_logo_compact.png"
html_static_path = ['_static', 'img']

master_doc = 'index'  # Ensure this points to your main document

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output