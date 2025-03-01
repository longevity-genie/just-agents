# syntax=docker/dockerfile:1

# Default to the CPU image (python:3.11-slim).
# To use the GPU image, override BASE_IMAGE at build time.
ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

### 1) Install system dependencies as root
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

### 2) Create a non-root user, with optional UID/GID
ARG UID
ARG GID
RUN if [ ! -z "$UID" ] && [ ! -z "$GID" ]; then \
    groupadd -g $GID appgroup && \
    useradd -u $UID -g appgroup -m -s /bin/bash appuser; \
    else \
    groupadd appgroup && \
    useradd -m -s /bin/bash -g appgroup appuser; \
    fi


### 3) Install Poetry (still as root)
RUN curl -sSL https://install.python-poetry.org | python3 -

### 4) Create app directory, copy project files in, fix ownership
WORKDIR /app
COPY . /app
RUN mkdir -p /app/models.d /app/tools /app/scripts /app/data

### 5) Copy entrypoint script (if not already copied) and make it executable
#    (Alternatively, you could have included this in step 4.)
COPY --chown=appuser:appgroup entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Make the /app directory owned by appuser (for any pre-copied files)
RUN chown -R appuser:appgroup /app

### 6) Install your main Python dependencies using Poetry (system-wide)
#    We disable virtualenv creation so Poetry installs into system site-packages.
RUN export POETRY_HOME="/opt/poetry" && \
    git config --global --add safe.directory /app && \
    git config --global --add safe.directory /app/core && \
    /root/.local/bin/poetry config virtualenvs.create false && \
    /root/.local/bin/poetry install --without dev

# Add verification step
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}')" || echo "PyTorch not found"

# Expose ports and set environment variables
EXPOSE 8088

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV APP_HOST="0.0.0.0"
ENV APP_PORT=8088
ENV AGENT_HOST="http://localhost"
ENV AGENT_PORT=8088
ENV AGENT_WORKERS=1
ENV AGENT_TITLE="Just-Agent endpoint"
ENV AGENT_PARENT_SECTION=""
ENV AGENT_DEBUG="true"
ENV AGENT_REMOVE_SYSTEM_PROMPT="false"
ENV AGENT_CONFIG="agent_profiles.yaml"
ENV TRAP_CHAT_NAMES="True"

# Declare the extra dependency argument (defaults to empty)
ARG EXTRA_DEPENDENCY=""

# Use Bash to add the extra dependency if it is provided.
RUN /bin/bash -c '\
    if [ -n "$EXTRA_DEPENDENCY" ]; then \
      echo "Adding extra dependency: $EXTRA_DEPENDENCY"; \
      /root/.local/bin/poetry config virtualenvs.create false; \
      /root/.local/bin/poetry add --no-interaction --no-cache "$EXTRA_DEPENDENCY"; \
    else \
      echo "No extra dependency provided"; \
    fi'

### 7) Switch to non-root user
USER appuser

# Ensure that ~/.local/bin is in PATH, so 'pip --user' installs are runnable
ENV PATH="/home/appuser/.local/bin:${PATH}"

#ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
# we will copy but not activate entrypoint script in the container as in docker-compose we will:
#services:
#    web-agent:
#      image: ghcr.io/longevity-genie/just-agents:main-gpu
#      entrypoint: /usr/local/bin/entrypoint.sh
CMD ["python", "-m", "just_agents.web.run_agent", "run-server-command"]