#!/bin/bash

# Exit on any error
set -e

# Check for --dry-run option
DRY_RUN=""
ALLOW_UNCOMMITTED=""
for arg in "$@"; do
    if [ "$arg" == "--dry-run" ]; then
        DRY_RUN="--dry-run"
        echo "Running in dry-run mode. No changes will be made."
    elif [ "$arg" == "--allow-uncommitted" ]; then
        ALLOW_UNCOMMITTED="true"
        echo "Warning: Allowing publishing with uncommitted changes."
    fi
done

# Disable keyring to avoid DBus/SecretService issues
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
REPO_URL="https://github.com/longevity-genie/just-agents.git"
    
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

    if ! poetry publish --skip-existing $DRY_RUN; then
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

# Verify all packages have consistent versions and clean them
base_version=$(poetry version -s | sed -E 's/([0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9.]+)?).*/\1/')
echo "Publishing version $base_version"

# Add check for PyPI version consistency
check_pypi_version() {
    local package=$1
    local expected_version=$2
    
    # Handle meta-package name differently
    local pkg_name=$([ "$package" != "." ] && echo "just-agents-${package}" || echo "just-agents")
    
    # Get the latest version from PyPI
    pypi_version=$(pip index versions "$pkg_name" 2>/dev/null | grep -m1 'Available versions:' | grep -oE '[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9.]+)?' | head -n1 || echo "0.0.0")
    
    # Compare versions using 'sort -V' for proper semantic versioning comparison
    if [ "$(printf '%s\n%s\n' "$pypi_version" "$expected_version" | sort -V | head -n1)" != "$pypi_version" ]; then
        echo "Error: PyPI version ($pypi_version) is ahead of local version ($expected_version) for $pkg_name"
        return 1
    fi
    return 0
}

# Also check the meta-package
if ! check_pypi_version "." "$base_version"; then
    echo "Version check failed for just-agents meta-package. Please ensure your local version is up to date with PyPI."
    exit 1
fi

# Verify PyPI versions before publishing
echo "Verifying PyPI versions..."
for pkg in "core" "tools" "coding" "web" "router" "examples" "."; do
    echo "Checking $pkg"
    if [ $(cd $pkg && poetry version -s) \< "$base_version" ]; then
        echo "Subpackage $pkg version is behind."
        exit 1
    fi
    if ! check_pypi_version "$pkg" "$base_version"; then
        echo "Version check failed. Please ensure your local version is up to date with PyPI or newer."
        exit 1
    fi
    echo "Subpackage $pkg version is up to date with PyPI or newer"
done

# Verify all packages have consistent versions
for pkg in "tools" "coding" "web" "router" "examples" "."; do
    pkg_version=$(cd $pkg && poetry version -s)
    if [ "$pkg_version" != "$base_version" ]; then
        echo "Error: Version mismatch in $pkg ($pkg_version != $base_version)"
        exit 1
    fi
done

# Verify GitHub repository version
check_github_version() {
    local expected_version=$1
    
    # Get the latest version tag from GitHub (stripping 'v' prefix)
    echo "Checking GitHub repository version..."
    # Corrected command to get the latest version tag without 'v' prefix
    github_version=$(git ls-remote --tags $REPO_URL | awk -F/ '{print $3}' | grep -v '{}' | sort -V | tail -n1 | sed 's/^v//')

    # Handle case where no versions exist yet
    if [ -z "$github_version" ]; then
        github_version="0.0.0"
    fi
    
    if [ "$github_version" \> "$expected_version" ]; then
        echo "Error: GitHub version (v$github_version) is ahead of local version ($expected_version)"
        return 1
    fi
    
    if [ "$github_version" != "$expected_version" ]; then
        echo "Error: GitHub version (v$github_version) does not match local version ($expected_version)"
        echo "Hint: Did you forget to create a git tag and push it?"
        return 1
    fi
    
    return 0
}

# Function to normalize URLs for comparison
normalize_url() {
    local url=$1
    # Remove .git suffix if present
    url=${url%.git}
    # Convert SSH URL to HTTPS format for comparison
    if [[ $url == git@* ]]; then
        # Convert SSH format to HTTPS and fix path separator
        url=$(echo "$url" | sed -e 's|^git@|https://|' -e 's|:\([^/]\)|/\1|')
    fi
    echo "$url"
}

# Function to check if the local repository is clean, up-to-date, and on the main branch
check_git_status() {
    # Ensure we are on the main branch
    local_branch=$(git rev-parse --abbrev-ref HEAD)
    if [ "$local_branch" != "main" ]; then
        echo "Error: You are not on the main branch. Please switch to the main branch before publishing."
        return 1
    fi

    # Normalize the REPO_URL for comparison
    normalized_repo_url=$(normalize_url "$REPO_URL")
    echo "Normalized REPO_URL: $normalized_repo_url"  # Debug statement

    # Find the remote that matches the normalized REPO_URL
    matching_remote=""
    while read -r name url rest; do
        echo "Processing remote: $name with URL: $url" >&2
        normalized_url=$(normalize_url "$url")
        if [ "$normalized_url" = "$normalized_repo_url" ]; then
            matching_remote="$name"
            echo "Matching remote found: $matching_remote"
            break
        fi
    done < <(git remote -v | grep "(fetch)")

    if [ -z "$matching_remote" ]; then
        echo "Error: No remote found matching the repository URL ($REPO_URL)."
        echo "Expected URL: $normalized_repo_url" >&2
        echo "Available remotes:" >&2
        git remote -v >&2
        return 1
    fi

    # Check for uncommitted changes
    if ! git diff-index --quiet HEAD --; then
        if [ -z "$ALLOW_UNCOMMITTED" ]; then
            echo "Error: Uncommitted changes found. Please commit or stash them before publishing."
            echo "Or use --allow-uncommitted flag to override this check."
            return 1
        else
            echo "Warning: Uncommitted changes detected, but continuing due to --allow-uncommitted flag."
        fi
    fi

    # Check if local branch is up-to-date with the correct remote
    git fetch "$matching_remote" main

    local_commit=$(git rev-parse HEAD)
    remote_commit=$(git rev-parse "$matching_remote/main")

    if [ "$local_commit" != "$remote_commit" ]; then
        echo "Error: Local main branch is not up-to-date with $matching_remote/main. Please pull the latest changes."
        return 1
    fi

    return 0
}

# Verify GitHub repository version and local git status
echo "Verifying GitHub repository version and local git status..."
if ! check_github_version "$base_version" || ! check_git_status; then
    echo "Version check or git status verification failed. Please ensure:"
    echo "1. You are on the main branch."
    echo "2. All changes are committed and pushed to the main branch."
    echo "3. CI tests have passed."
    echo "4. You've created and pushed a tag: git tag v$base_version && git push origin v$base_version"
    exit 1
fi

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