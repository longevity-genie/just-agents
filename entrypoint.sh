#!/usr/bin/env bash
set -e

ENV_KEYS_PATH=${ENV_KEYS_PATH:-"env/.env.keys"}

# If you have a "requirements.txt" in a mounted volume, you can install it here:
if [ -f "/app/tools/requirements.txt" ]; then
    echo "Installing additional dependencies from /app/tools/requirements.txt..."
    pip install --no-cache-dir -r /app/tools/requirements.txt
fi

# Check and run initialization scripts if they exist
if [ -f "/app/scripts/init.sh" ]; then
    echo "Running initialization script /app/scripts/init.sh..."
    bash /app/scripts/init.sh
elif [ -f "/app/scripts/init.py" ]; then
    echo "Running initialization script /app/scripts/init.py..."
    python /app/scripts/init.py
fi

echo "Waiting for keys file ${ENV_KEYS_PATH} to be created..."
while [ ! -f "$ENV_KEYS_PATH" ]; do
  sleep 1
done

echo "${ENV_KEYS_PATH} found! Starting service..."
# Finally, run the main container command (passed as CMD in the Dockerfile).
exec "$@"
