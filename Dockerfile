# Prism AI Gateway — one image runs the gateway and both mock providers.
# Build:  docker build -t prism-ai-gateway .
# Run:    docker compose up   (starts gateway + two mock providers)

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    XDG_CACHE_HOME=/app/.cache

WORKDIR /app

# Runtime dependencies only; requirements-dev.txt is not needed inside the image.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Application code, providers, console, and data (seed keys, gateway configs).
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY console/ ./console/
COPY data/ ./data/

# Run as a non-root user. /app/store keeps the SQLite file so it can be
# mounted as a volume without shadowing the configs in /app/data.
RUN useradd --create-home --uid 10001 prism \
    && mkdir -p /app/.cache /app/store \
    && chown -R prism:prism /app
USER prism

ENV PRISM_DATABASE_URL=sqlite+aiosqlite:////app/store/prism.db

EXPOSE 8080

# Overridden per service in docker-compose.yml (mock providers use
# scripts/mock_provider.py instead of the gateway).
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
