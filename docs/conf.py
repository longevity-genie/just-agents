# Configuration file for Sphinx documentation builder

project = 'just-agents'
copyright = '2024, Longevity Genie Team'
author = 'Longevity Genie Team'

# Add any Sphinx extension module names here
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'myst_parser',
    'sphinx.ext.intersphinx'
]

# Add after the existing extensions
autodoc_mock_imports = [
    'litellm',
    'pydantic',
    'requests',
    'numpydoc',
    'rich',
    'just_agents'
]

# Theme settings
html_theme = 'sphinx_rtd_theme'

# Add support for Markdown files
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# Add any paths that contain templates here, relative to this directory
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The master toctree document
master_doc = 'index'

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
}