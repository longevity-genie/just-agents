# Configuration file for Sphinx documentation builder

project = 'just-agents'
copyright = '2024, Longevity Genie Team'
author = 'Longevity Genie Team'

# Add any Sphinx extension module names here
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'myst_parser'
]

# Theme settings
html_theme = 'sphinx_rtd_theme'

# Add support for Markdown files
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}