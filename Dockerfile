FROM python:3.13-slim-bookworm AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential curl && rm -rf /var/lib/apt/lists/*

# Copy project files needed for install
COPY pyproject.toml poetry.lock LICENSE README.md /app/

# Install build backend and dependencies
RUN pip install --upgrade pip setuptools wheel hatch

# Install dependencies (main only)
RUN pip install --no-cache-dir .

# Copy source code
COPY src /app/src
COPY scripts /app/scripts

CMD ["python", "scripts/run_bot.py"]
