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

### 2) Create a non-root user, with UID/GID passed in at build time
#    The defaults here (1000:1000) can be overridden by docker build arguments
ARG UID=1000
ARG GID=1000
RUN groupadd -g $GID appgroup \
 && useradd -u $UID -g appgroup -m -s /bin/bash appuser


### 3) Install Poetry (still as root)
# Check if conda Python exists and set Python path accordingly
RUN if [ -f "/opt/conda/bin/python" ]; then \
    echo "Using conda Python" && \
    export PATH="/opt/conda/bin:${PATH}" && \
    curl -sSL https://install.python-poetry.org | /opt/conda/bin/python3 - ; \
    else \
    echo "Using system Python" && \
    curl -sSL https://install.python-poetry.org | python3 - ; \
    fi

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
RUN if [ -f "/opt/conda/bin/python" ]; then \
    # Conda-specific configuration
    export PATH="/opt/conda/bin:${PATH}" && \
    export PYTHON=$(which python) && \
    export POETRY_PYTHON=$(which python); \
else \
    # Non-conda configuration
    export POETRY_HOME="/opt/poetry"; \
fi && \
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

### 7) Switch to non-root user
USER appuser

# Ensure that ~/.local/bin is in PATH, so 'pip --user' installs are runnable
ENV PATH="/home/appuser/.local/bin:${PATH}"

# Add conda to PATH if it exists
RUN if [ -f "/opt/conda/bin/python" ]; then \
    echo "export PATH=/opt/conda/bin:\$PATH" >> /home/appuser/.bashrc ; \
    fi

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
# we will copy but not activate entrypoint script in the container as in docker-compose we will:
#services:
#    web-agent:
#      image: ghcr.io/longevity-genie/just-agents:main-gpu
#      entrypoint: /usr/local/bin/entrypoint.sh
CMD ["python", "-m", "just_agents.web.run_agent"]
