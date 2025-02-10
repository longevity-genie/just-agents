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

# Fix permissions for mounted volumes
chown -R appuser:appgroup /app /app/data /app/logs /app/env /app/models.d || true

# Switch to non-root user and execute the command
exec gosu appuser "$@"

# Handle additional dependencies safely via Poetry if requirements.txt is present
if [ -f "/app/tools/requirements.txt" ]; then
    echo "Installing additional dependencies from /app/tools/requirements.txt using Poetry..."
    su -c "poetry add $(grep -v '^#' /app/tools/requirements.txt)" appuser
fi

# Run initialization scripts if they exist
elif [ -f "/app/scripts/init.py" ]; then
    echo "Running initialization script /app/scripts/init.py..."
    python /app/scripts/init.py
fi
