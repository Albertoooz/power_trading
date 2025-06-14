# Use official Python 3.13 slim image as base
FROM python:3.13-slim-bookworm AS base

# Set environment variables for Python
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1

# Set working directory inside container
WORKDIR /app

# Install dependencies needed for poetry and building wheels
RUN apt-get update && apt-get install -y curl build-essential && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Add Poetry to PATH
ENV PATH="/root/.local/bin:$PATH"

# Copy only poetry config first for better caching
COPY poetry.lock pyproject.toml /app/

# Install dependencies
RUN poetry install --no-root --only main

# Copy application source code
COPY src /app/src
COPY scripts /app/scripts

# Default command to run your main trading script
CMD ["python", "scripts/run_bot.py"]
