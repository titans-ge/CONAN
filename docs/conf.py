# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../'))

#copy notebooks to tutorials folder for sphinx to find them
print("\nCopying notebooks to tutorial folder...")
os.system("cp ../Notebooks/TOI-216/TOI-216_TTV_fit.ipynb tutorial/")
os.system("cp ../Notebooks/TOI469/TOI469_multiplanet_LC_RV.ipynb tutorial/")
os.system("cp ../Notebooks/KELT-20/kelt-20_cheops_roll_compare.ipynb tutorial/")
os.system("cp ../Notebooks/WASP-127/WASP127_RV/CONAN_WASP-127_RV_tutorial.ipynb tutorial/")
os.system("cp ../Notebooks/WASP-127/WASP-127_EULER_LC/CONAN_WASP127_EULER.ipynb tutorial/")
os.system("cp ../Notebooks/WASP-127/WASP127_LC_RV/CONAN_WASP-127_LC_RV_tutorial.ipynb tutorial/")
#----------------


project = 'CONAN'
copyright = '2025, B. Akinsanmi, M. Lendl'
author = 'B. Akinsanmi, M. Lendl'
# version = CONAN.__version__
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
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinx.ext.autosummary",
    "nbsphinx",
    'myst_parser',
    # "myst_nb",
    "IPython.sphinxext.ipython_console_highlighting",
    'autoapi.extension'
    ]

autoapi_dirs = ["../CONAN"]
autoapi_ignore = ["*_version*", "*/types*"]
autoapi_options = [
    "members",
    "undoc-members",
    "private-members",
    "show-inheritance",
    "show-module-summary",
    # "special-members",
    # "imported-members",
]
autoapi_template_dir   = "_autoapi_templates"
suppress_warnings      = ["autoapi.python_import_resolution"]
autoapi_own_page_level = "method"
autoapi_member_order   = 'groupwise'
templates_path         = ['_templates']
exclude_patterns       = ['_build', 'Thumbs.db', '.DS_Store', "_autoapi_templates"]



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'sphinx_rtd_theme'
html_theme           = 'sphinx_book_theme'
html_static_path     = ['_static']
html_js_files = [
    'https://cdn.jsdelivr.net/npm/marked/marked.min.js',
    'gurubase-widget.js']
html_show_sourcelink = True
html_theme_options   = {
                        "path_to_docs": "docs",
                        "repository_url": "https://github.com/titans-ge/CONAN",
                        "repository_branch": "main",
                        "launch_buttons": {
                            "binderhub_url": "https://mybinder.org",
                            "notebook_interface": "jupyterlab",
                            },
                        "use_edit_page_button": True,
                        "use_issues_button": True,
                        "use_repository_button": True,
                        "use_download_button": True,
                        "use_sidenotes": True,
}


# # Concatenates classes docstrings with the ones from the __init__
# autoclass_content = 'class'
# autodoc_class_signature = "separated"

# html_theme_options = {
#     'navigation_depth': 5,
# }
# autosummary_generate = True

# autodoc_default_options = {
#     'members': True,
#     'member-order': 'bysource',
#     "private-members":False,
#     # 'special-members': '__init__',
#     'undoc-members': False,
#     'exclude-members': '__weakref__',
#     "show-inheritance": True
# }

# numpydoc_show_class_members=False

# def setup(app):
#     # https://stackoverflow.com/a/43186995
#     app.add_css_file('my_theme.css')

nb_execution_mode = "cache"
nb_execution_excludepatterns = []
nb_execution_timeout = -1