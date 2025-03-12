# syntax=docker/dockerfile:1

FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    gosu \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN export POETRY_VERSION="2.1.1" && \
    mkdir -p /opt/poetry && \
    curl -sSL https://install.python-poetry.org | POETRY_HOME=/opt/poetry python3 - && \
    ln -s /opt/poetry/bin/poetry /usr/local/bin/poetry && \
    export PATH="/opt/poetry/bin:$PATH" && \
    echo 'export PATH="/opt/poetry/bin:$PATH"' >> /etc/profile && \
    echo 'export PATH="/opt/poetry/bin:$PATH"' >> /etc/bash.bashrc && \
    export POETRY_HOME="/opt/poetry" 

# Add Poetry to PATH
ENV POETRY_HOME="/opt/poetry"
ENV PATH="/opt/poetry/bin:$PATH"

# Create necessary directories
RUN export CONTAINER_DIRS="/app /app/models.d /app/agent_tools /app/scripts /app/data /app/logs /app/tmp /app/env" && \
    mkdir -p $CONTAINER_DIRS && \
    chmod 777 $CONTAINER_DIRS

# Set up the app directory
WORKDIR /app
COPY ./web/containers/chat-ui-agents/pyproject.t__l /app/pyproject.toml
COPY ./LICENSE /app/LICENSE
COPY ./README.md /app/README.md

# Copy entire codebase
COPY . /opt/just-agents

# Install dependencies
RUN poetry config virtualenvs.create false && \
    poetry add "langfuse>=2.59.2" "opik>=1.4.12" "pandas>=2.2" --lock && \
    poetry install --no-interaction --no-cache --compile && \
    poetry show --tree

# Copy entrypoint script and make it executable
COPY ./web/containers/chat-ui-agents/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod 777 /app -R && \
    echo '#!/usr/bin/env bash' >> /app/init.py && \
    echo 'print("Preparation complete!")' >> /app/init.py && \
    chmod 777 /usr/local/bin/entrypoint.sh /app/init.py

# Expose ports and set environment variables
EXPOSE 8088

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# WebAgentConfig environment variables
ENV APP_HOST="0.0.0.0"
ENV APP_PORT=8089
ENV AGENT_WORKERS=1
ENV AGENT_TITLE="Just-Agent endpoint"
ENV AGENT_SECTION=""
ENV AGENT_PARENT_SECTION=""
ENV AGENT_FAILFAST="true"
ENV AGENT_DEBUG="true"
ENV AGENT_REMOVE_SYSTEM_PROMPT="false"
ENV AGENT_CONFIG_PATH="agent_profiles.yaml"
ENV ENV_KEYS_PATH="env/.env.local"
ENV APP_DIR="/app"
ENV TMP_DIR="tmp"
ENV LOG_DIR="logs"
ENV DATA_DIR="data"

# ChatUIAgentConfig additional environment variables (aligned with config.py)
ENV MODELS_DIR="models.d"
ENV ENV_MODELS_PATH="env/.env.local"
ENV REMOVE_DD_CONFIGS="true"
ENV TRAP_CHAT_NAMES="true"
ENV AGENT_HOST="http://127.0.0.1"
ENV AGENT_PORT=8088
ENV JSON_FILE_PATTERN="[0123456789][0123456789]_*.json"

# Keep running as root - let entrypoint handle user switching
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["python", "-m", "just_agents.web.run_agent", "run-chat-ui-server-command"]