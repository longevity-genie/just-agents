#!/bin/bash

# Exit on any error
set -e

# Load environment variables from .env file
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
elif [ -f ../.env ]; then
    export $(grep -v '^#' ../.env | xargs)
else
    echo "Error: .env file not found"
    exit 1
fi

# Function to build and publish a package
publish_package() {
    local dir=$1
    echo "Publishing $dir..."
    cd $dir
    
    # Verify package can be built before attempting publish
    if ! poetry build; then
        echo "Error: Failed to build $dir"
        cd ..
        return 1
    fi
    
    if ! poetry publish --skip-existing; then
        echo "Error: Failed to publish $dir"
        cd ..
        return 1
    fi
    
    cd ..
    return 0
}

# Check if PyPI token is configured
if [ -z "$PYPI_TOKEN" ]; then
    echo "Error: PyPI token not configured. Please set PYPI_TOKEN in your .env file."
    exit 1
fi

poetry config pypi-token.pypi $PYPI_TOKEN

# Verify all packages have consistent versions
version=$(cd core && poetry version -s)
echo "Publishing version $version"

for pkg in "tools" "coding" "web" "router" "examples"; do
    pkg_version=$(cd $pkg && poetry version -s)
    if [ "$pkg_version" != "$version" ]; then
        echo "Error: Version mismatch in $pkg ($pkg_version != $version)"
        exit 1
    fi
done

# Publish in the correct order based on dependencies
echo "Publishing packages in order..."

# First publish core since it's a dependency for all others
if ! publish_package "core"; then
    echo "Failed to publish core package. Aborting."
    exit 1
fi

# Then publish the components that depend on core
for package in "tools" "coding" "web" "router"; do
    if ! publish_package $package; then
        echo "Failed to publish $package package. Aborting."
        exit 1
    fi
done