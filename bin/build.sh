#!/bin/bash

# Get output directory from first argument or use default
WHEELS_DIR="${1:-$(pwd)/dist/}"
mkdir -p "$WHEELS_DIR"

# Function to build a package
build_package() {
    local dir=$1
    echo "Building $dir..."
    if [ ! -d "$dir" ]; then
        echo "Error: Directory $dir does not exist"
        exit 1
    fi
    if [ ! -f "$dir/pyproject.toml" ]; then
        echo "Error: No pyproject.toml found in $dir"
        exit 1
    fi
    rm -fr "$dir/dist"
    cd "$dir" || exit 1
    poetry lock || exit 1
    
    # Build directly to the central wheels directory
    poetry build --output "$WHEELS_DIR" || exit 1
    rm -f poetry.lock
}

# Ensure we're in the root directory
if [ ! -f "pyproject.toml" ]; then
    echo "Error: Script must be run from project root directory"
    exit 1
fi

echo "Building packages in order..."
# Store the current directory
CURRENT_DIR=$(pwd)

# Backup and modify pyproject.toml
#echo "Modifying pyproject.toml for meta-package build..."
#cp pyproject.toml pyproject.toml.bak
#sed -i 's/package-mode = false/packages = \[{include = "just_agents"}\]/' pyproject.toml

# Create temporary __init__.py
echo "Creating temporary __init__.py..."
touch "./just_agents/__init__.py"



# Build packages in a simplified order - meta package first, then subpackages
for package in "./" "core" "tools" "coding" "web" "router" "examples"; do
    build_package "$package"
    cd "$CURRENT_DIR" || exit 1
done

# Clean up temporary __init__.py
echo "Cleaning up temporary __init__.py..."
rm "./just_agents/__init__.py"

poetry lock

# Verify all wheels were copied
echo "Verifying all wheels in $WHEELS_DIR..."
ls -la "$WHEELS_DIR"

echo "All packages built!"