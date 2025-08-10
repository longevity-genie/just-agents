"""Namespace package for just_agents (metapackage portion).

This ensures that sibling distributions like `just_agents.core`,
`just_agents.web`, `just_agents.tools`, etc. are merged under a
single `just_agents` namespace regardless of import path order.
"""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)  # type: ignore[name-defined]


