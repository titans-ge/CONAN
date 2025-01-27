# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../'))

project = 'CONAN'
copyright = '2023, B. Akinsanmi, M. Lendl'
author = 'B. Akinsanmi, M. Lendl'
root_doc = 'index'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    "sphinx_autodoc_typehints",
    "sphinx.ext.autosummary",
    "nbsphinx"
    ]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'sphinx_rtd_theme'
html_theme = 'sphinx_book_theme'

html_static_path = ['_static']


# Concatenates classes docstrings with the ones from the __init__
autoclass_content = 'class'
autodoc_class_signature = "separated"

html_theme_options = {
    'navigation_depth': 5,
}
# autosummary_generate = True

autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    "private-members":False,
    # 'special-members': '__init__',
    'undoc-members': False,
    'exclude-members': '__weakref__',
    "show-inheritance": True
}

numpydoc_show_class_members=False

def setup(app):
    # https://stackoverflow.com/a/43186995
    app.add_css_file('my_theme.css')