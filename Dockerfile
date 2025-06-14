FROM python:3.13-slim-bookworm AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# 1. Zainstaluj systemowe pakiety potrzebne do budowy
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libffi-dev \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 2. Skopiuj pliki konfiguracyjne potrzebne do instalacji
COPY pyproject.toml poetry.lock LICENSE README.md /app/

# 3. Zaktualizuj pip i zainstaluj hatchling build backend
RUN pip install --upgrade pip setuptools wheel hatchling

# 4. Zainstaluj pakiet (główne zależności)
RUN pip install --no-cache-dir .

# 5. Skopiuj resztę kodu źródłowego
COPY src /app/src
COPY scripts /app/scripts

# 6. Domyślny CMD - uruchomienie bota
CMD ["python", "scripts/run_bot.py"]
