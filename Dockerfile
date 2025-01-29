# Use Python 3.10 as base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Create writable and mountable directories
RUN mkdir -p /app/models.d /app/tools && chmod -R 777 /app/models.d /app/tools

# Mark them as volumes to ensure they remain writable even when mounted
VOLUME ["/app/models.d", "/app/tools"]

# Copy project files
COPY . .

# Configure poetry to not create virtual environment
RUN /root/.local/bin/poetry config virtualenvs.create false

# Install dependencies
RUN /root/.local/bin/poetry install --without dev

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Set default environment variables (can be overridden in docker-compose)
ENV APP_HOST="0.0.0.0"
ENV APP_PORT=8088

EXPOSE 8088

ENV AGENT_HOST="http://localhost"
ENV AGENT_PORT=8088
ENV AGENT_WORKERS=1
ENV AGENT_TITLE="Just-Agent endpoint"
ENV AGENT_SECTION=""
ENV AGENT_PARENT_SECTION=""
ENV AGENT_DEBUG="true"
ENV AGENT_REMOVE_SYSTEM_PROMPT="false"
ENV AGENT_CONFIG="agent_profiles.yaml"

# Default command
CMD ["python", "-m", "just_agents.web.run_agent"] 