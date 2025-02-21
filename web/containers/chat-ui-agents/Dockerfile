# syntax=docker/dockerfile:1

ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

### 1) Install system dependencies as root
RUN apt-get update && apt-get install -y \
    git \
    curl \
    gosu \
    && rm -rf /var/lib/apt/lists/*

### 2) Install Poetry (still as root)
RUN export POETRY_VERSION="2.1.1" && \
    mkdir -p /opt/poetry && \
    curl -sSL https://install.python-poetry.org | POETRY_HOME=/opt/poetry python3 - && \
    ln -s /opt/poetry/bin/poetry /usr/local/bin/poetry && \
    # Add poetry to system-wide PATH and for new users
    echo 'export PATH="/opt/poetry/bin:$PATH"' >> /etc/profile && \
    echo 'export PATH="/opt/poetry/bin:$PATH"' >> /etc/bash.bashrc && \
    export POETRY_HOME="/opt/poetry" && \
    export PATH="/opt/poetry/bin:$PATH"

### 3) Create app directory, copy project files in
WORKDIR /app
COPY . /app
RUN mkdir -p /app/models.d /app/agent_tools /app/scripts /app/data /app/logs /app/tmp

### 4) Copy entrypoint script and make it executable
COPY ./web/containers/chat-ui-agents/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN echo '#!/usr/bin/env bash' >> /app/init.py
RUN echo 'print("Preparation complete!")' >> /app/init.py
RUN chmod +x /usr/local/bin/entrypoint.sh
RUN chmod +x /app/init.py

### 5) Install main Python dependencies
RUN poetry config virtualenvs.create false && \
    poetry install --without dev && \
    poetry add langfuse>=2.59.2 opik>=1.4.12 numpy>=2.2 just-agents-tools

### 6) Expose ports and set environment variables
EXPOSE 8088

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# WebAgentConfig environment variables
ENV APP_HOST="0.0.0.0"
ENV APP_PORT=8088
ENV AGENT_WORKERS=1
ENV AGENT_TITLE="Just-Agent endpoint"
ENV AGENT_SECTION=""
ENV AGENT_PARENT_SECTION=""
ENV AGENT_DEBUG="true"
ENV AGENT_REMOVE_SYSTEM_PROMPT="false"
ENV AGENT_CONFIG_PATH="agent_profiles.yaml"
ENV ENV_KEYS_PATH="env/.env.local"

# ChatUIAgentConfig additional environment variables
ENV ENV_MODELS_PATH="env/.env.local"
ENV MODELS_DIR="models.d"
ENV AGENT_FAILFAST="true"
ENV REMOVE_DD_CONFIGS="true"
ENV TRAP_CHAT_NAMES="true"
ENV AGENT_HOST="http://127.0.0.1"
ENV AGENT_PORT=8088
ENV JSON_FILE_PATTERN="[0123456789][0123456789]_*.json"

### 7) Keep running as root - let entrypoint handle user switching
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

CMD ["python", "-m", "just_agents.web.run_agent", "run-chat-ui-server-command"]