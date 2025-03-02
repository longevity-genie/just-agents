#!/bin/bash
script_dir=$(dirname "$(readlink -f "$0")")
project_root=$(realpath "${script_dir}/../../../")

# Run the project's build script first
#echo "Running project build script..."
#${project_root}/bin/build.sh

# Build the Docker container
echo "Building Docker container..."
docker build -f ${script_dir}/Dockerfile -t ghcr.io/longevity-genie/just-agents/chat-ui-agents:local ${project_root}

# Pull into podman
echo "Pulling into podman..."
podman pull docker-daemon:ghcr.io/longevity-genie/just-agents/chat-ui-agents:local
