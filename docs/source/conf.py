# Configuration file for the Sphinx documentation builder.

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))  # Adjust the path as needed

project = 'clearbox-synthetic-kit'
copyright = '2025, Clearbox-AI'
author = 'Clearbox-AI'

html_js_files = [
    'mathjax.js',  # Manually include MathJax
]

extensions = [
    'sphinx.ext.autodoc',  # Automatically document code
    'sphinx.ext.napoleon',  # Support for Google-style and NumPy-style docstrings
    'sphinx.ext.viewcode',  # Add links to source code
    'sphinx.ext.intersphinx',  # Link to other project's documentation
    'sphinx.ext.mathjax', # Enables MathJax for rendering LaTeX equations
    'myst_parser', # To include Markdown files - If not installed when running 'make html' run -> pip install myst-parser - Add it in requirements.txt
    'sphinx_rtd_theme'
]

myst_enable_extensions = [
    "html_image", # Allows html images format conversion
    "dollarmath",  # Enables $...$ and $$...$$ syntax for math
    "amsmath", # Enables support for amsmath-style math blocks
]
myst_links_external_new_tab = True

mathjax3_config = {
    'tex': {
        'inlineMath': [['$', '$'], ['\\(', '\\)']],
        'displayMath': [['$$', '$$'], ['\\[', '\\]']],
    }
}

source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}

templates_path = ['_templates']
exclude_patterns = ['diffusion.py',
                    'engine.py',
                    'gower_matrix_c.pyx',
                    '**/VAE/*',
                    '**/transformers/*']

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "logo_only": True,
    # "style_nav_header_background": "#483a8f",
    "prev_next_buttons_location": None  # Removes "Previous" and "Next" buttons
}

html_static_path = ['_static', 'img']

html_css_files = [
    'style.css',
]

html_logo = "img/synthetickit_compatto 02.png"
html_favicon = "img/favicon.ico"

master_doc = 'index'  # Ensure this points to your main document

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output