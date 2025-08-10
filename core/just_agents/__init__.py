"""Namespace package for just_agents.

This enables merging subpackages provided by sibling distributions,
such as `just_agents.web`, `just_agents.tools`, etc.
"""

from pkgutil import extend_path

# Allow this package to be extended by other distributions that
# provide the same top-level package name.
__path__ = extend_path(__path__, __name__)  # type: ignore[name-defined]

