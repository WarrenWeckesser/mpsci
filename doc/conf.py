# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'mpsci'
copyright = '2019, Warren Weckesser'
author = 'Warren Weckesser'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
]

mathjax_path = "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_CHTML"


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'alabaster'
#html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    #'logo': 'logo.png',
    'github_user': 'WarrenWeckesser',
    'github_repo': 'mpsci',
    'github_button': False,
    'fixed_sidebar': True,
    'sidebar_collapse': True,
    'extra_nav_links': {'Index': 'genindex.html',
                        'Source code':
                            'https://github.com/WarrenWeckesser/mpsci'}
}

html_sidebars = {
    '**': [
        'about.html',
        'navigation.html',
        #'relations.html',
        'searchbox.html',
        #'donate.html',
    ]
}
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
#html_static_path = ['_static']
html_static_path = []

html_copy_source = False
html_show_sourcelink = False

# -----


# Does this work?
autosummary_generate = True
