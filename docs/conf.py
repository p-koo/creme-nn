# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'CREME'
copyright = '2023, Shushan Toneyan and Peter K Koo'
author = 'Shushan Toneyan and Peter K Koo'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'autoapi.extension',
]

autoapi_type = 'python'
autoapi_dirs = ['../creme']

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'tensorflow': ('https://www.tensorflow.org/versions/r2.11/api_docs/', None),
}
intersphinx_disabled_domains = ['std']


templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']

epub_show_urls = 'footnote'
