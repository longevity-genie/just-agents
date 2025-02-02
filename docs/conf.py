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
    'sphinx.ext.intersphinx',
    'sphinx_autodoc_typehints',
    'sphinx_autoapi'
]

# Add after the existing extensions
autodoc_mock_imports = [
    'litellm',
    'pydantic',
    'requests',
    'numpydoc',
    'rich'
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

# Enable markdown parsing with MyST
myst_enable_extensions = [
    "colon_fence",
    "deflist"
]

# AutoAPI settings
autoapi_type = 'python'
autoapi_dirs = ['../just_agents']
autoapi_add_toctree_entry = True
autoapi_options = ['members', 'undoc-members', 'show-inheritance', 'show-module-summary']
autoapi_python_use_implicit_namespaces = True