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

# Copy project files
COPY . .

# Configure poetry to not create virtual environment
RUN /root/.local/bin/poetry config virtualenvs.create false

# Install dependencies
RUN /root/.local/bin/poetry install --without dev

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python", "-m", "just_agents.web.run_agent"] 