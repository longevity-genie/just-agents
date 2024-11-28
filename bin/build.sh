#!/bin/bash

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
    cd "$dir" || exit 1
    poetry build || exit 1
    cd ..
}

# Ensure we're in the root directory
if [ ! -f "pyproject.toml" ]; then
    echo "Error: Script must be run from project root directory"
    exit 1
fi

echo "Building packages in order..."

# First build core since it's a dependency for all others
build_package "core"

# Then build the components that depend on core
for package in "tools" "coding" "web" "router" "examples"; do
    build_package "$package"
done

# Store the current directory
CURRENT_DIR=$(pwd)

# Backup and modify pyproject.toml
echo "Modifying pyproject.toml for meta-package build..."
cp pyproject.toml pyproject.toml.bak
sed -i 's/package-mode = false/packages = \[{include = "just_agents"}\]/' pyproject.toml

# Create temporary __init__.py
echo "Creating temporary __init__.py..."
touch "./just_agents/__init__.py"

# Build the meta-package
build_package "./"

# Restore original pyproject.toml
echo "Restoring original pyproject.toml..."
mv pyproject.toml.bak pyproject.toml

# Return to the original directory before cleanup
cd "$CURRENT_DIR" || exit 1

# Clean up temporary __init__.py
echo "Cleaning up temporary __init__.py..."
rm "./just_agents/__init__.py"

echo "All packages built!"