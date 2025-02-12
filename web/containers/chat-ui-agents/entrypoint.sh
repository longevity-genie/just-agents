#!/usr/bin/env bash
set -e

# Default USER_ID/GROUP_ID if not provided
USER_ID=${USER_ID:-1000}
GROUP_ID=${GROUP_ID:-1000}

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
        echo "Executing command: poetry add $REQS"
        
        # Run poetry add with the filtered dependencies
        if poetry add $REQS; then
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

# Run initialization script if it exists, using hash comparison to ensure idempotency
INIT_SCRIPT="/app/scripts/init.py"
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
        echo "--------------------------------"
        
        # Ensure the script is executable
        chmod +x "$INIT_SCRIPT"
        
        # Execute the initialization script as appuser
        if gosu appuser:appgroup python3 "$INIT_SCRIPT"; then
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

# Finally, switch to non-root user appuser and execute the provided command.
exec gosu appuser:appgroup "$@"
