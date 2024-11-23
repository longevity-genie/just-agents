#!/bin/bash
# we assume that the script is running from the root of the project
# like this: ./bin/publish.sh

# Add environment variable check
if [ -z "$PYPI_TOKEN" ]; then
    echo "Error: PYPI_TOKEN environment variable is not set"
    exit 1
fi

# Parse command line arguments
skip_publish=false
while getopts "s" opt; do
    case $opt in
        s) skip_publish=true ;;
        *) echo "Usage: $0 [-s]" >&2
           echo "  -s: Skip publishing to PyPI"
           exit 1 ;;
    esac
done

# Function to extract version from pyproject.toml
get_version() {
    if [ ! -f "$1/pyproject.toml" ]; then
        echo "Error: $1/pyproject.toml not found"
        exit 1
    fi
    grep 'version = ' "$1/pyproject.toml" | cut -d'"' -f2
}

# Check if directories exist
for pkg in core web tools coding router examples; do
    if [ ! -d "$pkg" ]; then
        echo "Error: Directory $pkg not found"
        exit 1
    fi
done

# Check versions
base_version=$(get_version "core")
for pkg in core web tools coding router examples; do
    pkg_version=$(get_version "$pkg")
    if [ "$base_version" != "$pkg_version" ]; then
        echo "Version mismatch: $pkg ($pkg_version) != core ($base_version)"
        exit 1
    fi
done

# Build each package
for pkg in core web tools coding router examples; do
    echo "Building just_agents/$pkg..."
    (cd "$pkg" && python -m build) || { echo "Failed to build $pkg"; exit 1; }
done

# Upload all packages if not skipped
if [ "$skip_publish" = false ]; then
    for pkg in core web tools coding router examples; do
        echo "Uploading just_agents/$pkg..."
        if [ -d "$pkg/dist" ]; then
            twine upload --verbose "$pkg/dist/*" --username "__token__" --password "$PYPI_TOKEN" || { echo "Failed to upload just_agents/$pkg"; exit 1; }
        else
            echo "Warning: No dist directory found for just_agents/$pkg"
        fi
    done
else
    echo "Skipping package upload (--skip-publish flag set)"
fi