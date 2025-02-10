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
RUN if [ -f "/opt/conda/bin/python" ]; then \
    echo "Using conda Python" && \
    export PATH="/opt/conda/bin:${PATH}" && \
    curl -sSL https://install.python-poetry.org | /opt/conda/bin/python3 - ; \
    else \
    echo "Using system Python" && \
    curl -sSL https://install.python-poetry.org | python3 - ; \
    fi

### 3) Create app directory, copy project files in
WORKDIR /app
COPY . /app
RUN mkdir -p /app/models.d /app/tools /app/scripts /app/data /app/logs

### 4) Copy entrypoint script and make it executable
COPY ./web/containers/chat-ui-agents/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN echo 'print("Preparation complete!")' > /app/scripts/init.py
RUN chmod +x /usr/local/bin/entrypoint.sh
RUN chmod +x /app/scripts/init.py

### 5) Install main Python dependencies
RUN if [ -f "/opt/conda/bin/python" ]; then \
    export PATH="/opt/conda/bin:${PATH}" && \
    export PYTHON=$(which python) && \
    export POETRY_PYTHON=$(which python); \
else \
    export POETRY_HOME="/opt/poetry"; \
fi && \
/root/.local/bin/poetry config virtualenvs.create false && \
/root/.local/bin/poetry install --without dev

# Add verification step
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}')" || echo "PyTorch not found"

### 6) Expose ports and set environment variables
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

### 7) Keep running as root - let entrypoint handle user switching
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

CMD ["python", "-m", "just_agents.web.run_agent", "run-chat-ui-server-command"]
CMD sh -c "python /app/scripts/prepare.py && python -m my_project.main"