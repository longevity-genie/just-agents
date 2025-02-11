#!/usr/bin/env bash
set -e

# Default UID/GID if not provided
USER_ID=${UID:-1000}
GROUP_ID=${GID:-1000}

# Ensure the group exists
if ! getent group appgroup >/dev/null; then
    groupadd --gid "$GROUP_ID" appgroup
fi

# Ensure the user exists
if ! id -u appuser >/dev/null 2>&1; then
    useradd --uid "$USER_ID" --gid appgroup --create-home appuser
fi

# Fix permissions for mounted volumes (ignore errors if paths are not present)
chown -R appuser:appgroup /app /app/data /app/logs /app/agent_tools /app/env /app/models.d /app/tmp || true
cd /app

# Handle additional dependencies safely via Poetry if requirements.txt is present
REQUIREMENTS_FILE="/app/agent_tools/requirements.txt"
INSTALL_FLAG="/app/tmp/.requirements_installed"

if [ -f "$REQUIREMENTS_FILE" ]; then
    if [ ! -f "$INSTALL_FLAG" ] || [ "$REQUIREMENTS_FILE" -nt "$INSTALL_FLAG" ]; then
        echo "Installing additional dependencies from $REQUIREMENTS_FILE using Poetry..."
        # Filter out comments and empty lines, then add all requirements at once
        if grep -v '^#' "$REQUIREMENTS_FILE" | grep -v '^[[:space:]]*$' | xargs poetry add; then
            # Create or update the flag file only if poetry install succeeded
            touch "$INSTALL_FLAG"
            echo "Dependencies installation completed successfully"
        else
            echo "Dependencies installation failed"
            exit 1
        fi
    else
        echo "Dependencies already installed, skipping..."
    fi
fi

# Run initialization scripts if they exist,
# executing them as appuser.
INIT_SCRIPT="/app/scripts/init.py"
INIT_FLAG="/app/tmp/.init_completed"

if [ -f "$INIT_SCRIPT" ]; then
    if [ ! -f "$INIT_FLAG" ] || [ "$INIT_SCRIPT" -nt "$INIT_FLAG" ]; then
        echo "Running initialization script $INIT_SCRIPT..."
        chmod +x "$INIT_SCRIPT"
        if gosu appuser:appgroup python3 "$INIT_SCRIPT"; then
            touch "$INIT_FLAG"
            echo "Initialization completed successfully"
        else
            echo "Initialization failed"
            exit 1
        fi
    else
        echo "Initialization already completed, skipping..."
    fi
fi

# Finally, switch to non-root user appuser and execute the provided command.
exec gosu appuser:appgroup "$@"
