#!/bin/bash
# we assume that the script is running from the root of the project
# like this: ./bin/publish.sh

# Function to extract version from pyproject.toml
get_version() {
    if [ ! -f "$1/pyproject.toml" ]; then
        echo "Error: $1/pyproject.toml not found"
        exit 1
    fi
    grep 'version = ' "$1/pyproject.toml" | cut -d'"' -f2
}

# Check if directories exist
for pkg in just_agents just_agents_web just_agents_tools just_agents_coding just_agents_router just_agents_examples; do
    if [ ! -d "$pkg" ]; then
        echo "Error: Directory $pkg not found"
        exit 1
    fi
done

# Check versions
base_version=$(get_version "just_agents")
for pkg in just_agents just_agents_web just_agents_tools just_agents_coding just_agents_router just_agents_examples; do
    pkg_version=$(get_version "$pkg")
    if [ "$base_version" != "$pkg_version" ]; then
        echo "Version mismatch: $pkg ($pkg_version) != just_agents ($base_version)"
        exit 1
    fi
done

# Build each package
for pkg in just_agents just_agents_web just_agents_tools just_agents_coding just_agents_router just_agents_examples; do
    echo "Building $pkg..."
    (cd "$pkg" && python -m build) || { echo "Failed to build $pkg"; exit 1; }
done

# Upload all packages
for pkg in just_agents just_agents_web just_agents_tools just_agents_coding just_agents_router just_agents_examples; do
    echo "Uploading $pkg..."
    if [ -d "$pkg/dist" ]; then
        twine upload --verbose "$pkg/dist/*" --config-file .pypirc || { echo "Failed to upload $pkg"; exit 1; }
    else
        echo "Warning: No dist directory found for $pkg"
    fi
done