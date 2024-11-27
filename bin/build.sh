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

# Finally build the meta-package
build_package "./"

echo "All packages built!"