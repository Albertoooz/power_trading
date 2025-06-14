# Use official Python 3.13 slim image as base
FROM python:3.13-slim-bookworm AS base

# Set environment variables for Python and Poetry
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    POETRY_VIRTUALENVS_CREATE=false \   # Disable Poetry virtualenv creation inside container
    POETRY_NO_INTERACTION=1             # Disable interactive prompts during Poetry install

# Set working directory inside container
WORKDIR /app

# Install system dependencies required for Poetry and building Python packages
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry using the official installer script
RUN curl -sSL https://install.python-poetry.org | python3 -

# Add Poetry to PATH for this container session
ENV PATH="/root/.local/bin:$PATH"

# Copy only the Poetry config files first to leverage Docker layer caching
COPY poetry.lock pyproject.toml /app/

# Install Python dependencies without installing the local package (--no-root),
# only main dependencies (not dev)
RUN poetry install --no-root --only main

# Copy the rest of your application code
COPY src /app/src
COPY scripts /app/scripts

# Default command: run your trading bot script
CMD ["python", "scripts/run_bot.py"]
