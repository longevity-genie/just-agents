#!/bin/bash

# Function to extract version from pyproject.toml
get_version() {
    grep 'version = ' "$1/pyproject.toml" | cut -d'"' -f2
}

# Check versions
base_version=$(get_version "just-agents")
for pkg in just-agents-web just-agents-tools just-agents-coding just-agents-all; do
    pkg_version=$(get_version "$pkg")
    if [ "$base_version" != "$pkg_version" ]; then
        echo "Version mismatch: $pkg ($pkg_version) != just-agents ($base_version)"
        exit 1
    fi
done

# Remove old distributions
rm -rf dist/
rm -rf */dist/
rm -rf *.egg-info
rm -rf */*.egg-info

# Build each package
cd just-agents && python -m build && cd ..
cd just-agents-web && python -m build && cd ..
cd just-agents-tools && python -m build && cd ..
cd just-agents-coding && python -m build && cd ..
cd just-agents-all && python -m build && cd ..

# Upload all packages
twine upload --verbose just-agents/dist/* --config-file .pypirc
twine upload --verbose just-agents-web/dist/* --config-file .pypirc
twine upload --verbose just-agents-tools/dist/* --config-file .pypirc
twine upload --verbose just-agents-coding/dist/* --config-file .pypirc
twine upload --verbose just-agents-all/dist/* --config-file .pypirc