FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       bash build-essential curl git openjdk-21-jre postgresql-client libpostgresql-jdbc-java \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md requirements.txt pytest.ini .env.example /app/
COPY coding_fgf /app/coding_fgf
COPY configs /app/configs
COPY scripts /app/scripts
COPY docs /app/docs
COPY prompts /app/prompts
COPY tests /app/tests

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -e ".[live,dev]"

CMD ["python", "-m", "coding_fgf", "--help"]
