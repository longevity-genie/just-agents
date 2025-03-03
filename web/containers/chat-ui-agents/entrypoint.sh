#!/usr/bin/env bash
set -e

# Default USER_ID/GROUP_ID if not provided
USER_ID=${USER_ID:-1000}
GROUP_ID=${GROUP_ID:-1000}

# Detect container runtime (prioritize cgroup, fallback to environment variable)
if grep -q podman /proc/self/cgroup /proc/1/cgroup || [ -f /run/.containerenv ] || [ "$container" = "podman" ]; then
    CONTAINER_RUNTIME="podman"
elif grep -q docker /proc/self/cgroup /proc/1/cgroup || [ -f /run/.dockerenv ] || [ "$container" = "docker" ]; then
    CONTAINER_RUNTIME="docker"
else
    CONTAINER_RUNTIME="unknown"
fi

echo "Detected container runtime: $CONTAINER_RUNTIME"

# Handle additional dependencies safely via Poetry if requirements.txt is present
REQUIREMENTS_FILE="/app/agent_tools/requirements.txt"
INSTALL_FLAG="/app/tmp/.requirements_installed"

# Check if the additional requirements file exists
if [ -f "$REQUIREMENTS_FILE" ]; then
    # Compute SHA256 hash of the current requirements.txt for comparison
    CURRENT_HASH=$(sha256sum "$REQUIREMENTS_FILE" | awk '{print $1}')

    # If the flag file exists, read the previously stored hash; otherwise, set an empty string
    if [ -f "$INSTALL_FLAG" ]; then
        INSTALLED_HASH=$(cat "$INSTALL_FLAG")
    else
        INSTALLED_HASH=""
    fi

    # Compare the current hash with the stored hash
    if [ "$CURRENT_HASH" != "$INSTALLED_HASH" ]; then
        echo "Installing additional dependencies from $REQUIREMENTS_FILE using Poetry..."

        # Display the contents of the requirements file for debugging
        echo "Contents of $REQUIREMENTS_FILE:"
        echo "--------------------------------"
        cat "$REQUIREMENTS_FILE"
        echo "--------------------------------"

        # Filter out comments and blank lines to obtain the list of dependencies
        REQS=$(grep -vE '^(#|[[:space:]]*$)' "$REQUIREMENTS_FILE")

        # Print out the full poetry command that will be executed for debugging purposes
        echo "Executing command: poetry config virtualenvs.create false && poetry add --no-interaction --no-cache $REQS"

        # Configure poetry to not use virtualenv and run poetry add with the filtered dependencies
        if poetry config virtualenvs.create false && poetry add --no-interaction --no-cache $REQS; then
            # On successful installation, update (or create) the flag file with the current hash
            echo "$CURRENT_HASH" > "$INSTALL_FLAG"
            echo "Dependencies installation completed successfully"
        else
            echo "Dependencies installation failed"
            exit 1
        fi
    else
        echo "Hash of '$REQUIREMENTS_FILE' file exists and has not changed: $CURRENT_HASH"
        echo "Dependencies already installed, skipping..."
    fi
fi

# Initialize GOSU variable
GOSU=""

# If running as root on Docker, perform setup and set GOSU to drop privileges.
if [ "$(id -u)" = "0" ] && [ "$CONTAINER_RUNTIME" != "podman" ]; then
    echo "Running as root on Docker. Setting up user, group, and chown operations..."

    # Ensure the group exists
    if ! getent group appgroup >/dev/null; then
        groupadd --gid "$GROUP_ID" appgroup
    fi

    # Ensure the user exists
    if ! id -u appuser >/dev/null 2>&1; then
        useradd --uid "$USER_ID" --gid appgroup --create-home appuser
    fi

    # Fix permissions for mounted volumes
    chown -R appuser:appgroup /app /app/data /app/logs /app/agent_tools /app/env /app/models.d /app/tmp

    # Set GOSU variable for subsequent commands
    GOSU="gosu appuser:appgroup"
else
    echo "Running in Podman rootless mode or already non-root. Skipping user creation."
    #chown -R $(id -u):$(id -g) /app /app/data /app/logs /app/agent_tools /app/env /app/models.d /app/tmp

fi

# Run initialization script if it exists, using hash comparison to ensure idempotency
INIT_SCRIPT="/app/init.py"
INIT_FLAG="/app/tmp/.init_completed"

if [ -f "$INIT_SCRIPT" ]; then
    # Compute SHA256 hash of the init script for comparison
    CURRENT_HASH=$(sha256sum "$INIT_SCRIPT" | awk '{print $1}')

    # If the flag file exists, read the stored hash; otherwise, default to an empty string
    if [ -f "$INIT_FLAG" ]; then
        INSTALLED_HASH=$(cat "$INIT_FLAG")
    else
        INSTALLED_HASH=""
    fi

    # Print the computed hash for debugging purposes
    echo "Current hash of $INIT_SCRIPT: $CURRENT_HASH"

    # Compare the current hash with the stored hash
    if [ "$CURRENT_HASH" != "$INSTALLED_HASH" ]; then
        echo "Running initialization script $INIT_SCRIPT..."

        # Display the contents of the initialization script for debugging purposes
        echo "20 lines of $INIT_SCRIPT:"
        echo "--------------------------------"
        cat "$INIT_SCRIPT" | head -n 20
        echo
        echo "--------------------------------"

        # Ensure the script is executable
        chmod +x "$INIT_SCRIPT"

        # Execute the initialization script as appuser
        if $GOSU python3 "$INIT_SCRIPT"; then
            # Upon successful execution, update or create the flag file with the current hash
            echo "$CURRENT_HASH" > "$INIT_FLAG"
            echo "Initialization completed successfully"
        else
            echo "Initialization failed"
            exit 1
        fi
    else
        echo "Hash of '$INIT_SCRIPT' file exists and has not changed: $CURRENT_HASH"
        echo "Initialization already completed, skipping..."
    fi
fi

# Finally, execute the provided command using GOSU if set.
echo "Executing command: $GOSU $*"
exec $GOSU "$@"

