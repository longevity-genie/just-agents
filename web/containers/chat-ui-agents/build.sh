#!/bin/bash
script_dir=$(dirname "$(readlink -f "$0")")
docker build -f ${script_dir}/Dockerfile -t ghcr.io/longevity-genie/just-agents/chat-ui-agents:local ${script_dir}/../../../