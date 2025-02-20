#!/bin/bash

# Exit on any error
set -e

# Disable keyring to avoid DBus/SecretService issues
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring

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

# Configure Poetry to use the PyPI token
poetry config pypi-token.pypi $PYPI_TOKEN

# Verify all packages have consistent versions and clean them
base_version=$(cd core && poetry version -s | sed -E 's/([0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9.]+)?).*/\1/')
echo "Publishing version $base_version"

# First set clean versions for all packages
for pkg in "core" "tools" "coding" "web" "router" "examples" "."; do
    echo "Setting clean version $base_version for $pkg"
    (cd $pkg && poetry version $base_version) || exit 1
done

# Verify all packages have consistent versions
for pkg in "tools" "coding" "web" "router" "examples" "."; do
    pkg_version=$(cd $pkg && poetry version -s)
    if [ "$pkg_version" != "$base_version" ]; then
        echo "Error: Version mismatch in $pkg ($pkg_version != $base_version)"
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
# Add just-agents to this list as well
for package in "tools" "coding" "web" "router" "examples"; do
    if ! publish_package $package; then
        echo "Failed to publish $package package. Aborting."
        exit 1
    fi
done

# Create temporary __init__.py for the meta-package build
echo "Creating temporary __init__.py..."
touch "./just_agents/__init__.py"

# Store the current directory
CURRENT_DIR=$(pwd)

# Temporarily modify pyproject.toml to use published versions and update package mode
echo "Updating dependencies and package mode in pyproject.toml..."
cp pyproject.toml pyproject.toml.bak
sed -i \
    -e "s|{ path = \"core\", develop = true }|\"$base_version\"|g" \
    -e "s|{ path = \"tools\", develop = true }|\"$base_version\"|g" \
    -e "s|{ path = \"coding\", develop = true }|\"$base_version\"|g" \
    -e "s|{ path = \"web\", develop = true }|\"$base_version\"|g" \
    -e "s|{ path = \"router\", develop = true }|\"$base_version\"|g" \
    -e "s|{ path = \"examples\", develop = true }|\"$base_version\"|g" \
    -e 's/package-mode = false/packages = [{include = "just_agents"}]/' \
    pyproject.toml

# Finally publish the meta-package
if ! publish_package "."; then
    cd "$CURRENT_DIR" || exit 1
    echo "Failed to publish just-agents meta-package. Aborting."
    # Restore original pyproject.toml
    mv pyproject.toml.bak pyproject.toml
    exit 1
fi

# Return to the original directory before cleanup
cd "$CURRENT_DIR" || exit 1

# Clean up temporary __init__.py
echo "Cleaning up temporary __init__.py..."
rm "./just_agents/__init__.py"

# Restore package mode in pyproject.toml
echo "Restoring original pyproject.toml..."
mv pyproject.toml.bak pyproject.toml