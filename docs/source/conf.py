# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Clearbox Engine'
copyright = '2024, Luca Gilli'
author = 'Luca Gilli'
release = '1.0.3'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',  # Automatically document code
    'sphinx.ext.napoleon',  # Support for Google-style and NumPy-style docstrings
    'sphinx.ext.viewcode',  # Add links to source code
    'sphinx.ext.intersphinx',  # Link to other project's documentation
    'myst_parser' # To include Markdown files - if not installed when running 'make html' run -> pip install myst-parser
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
