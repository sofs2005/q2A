# syntax=docker/dockerfile:1.7

# Stage 1: Build frontend assets once on the build platform.
FROM --platform=$BUILDPLATFORM node:20-bookworm-slim AS frontend-builder
WORKDIR /app
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Stage 2: Runtime image.
FROM python:3.12-slim-bookworm
WORKDIR /workspace

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONIOENCODING=utf-8 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860 \
    WORKERS=1 \
    LOG_LEVEL=INFO \
    PYTHONPATH=/workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

COPY backend/ ./backend/
COPY start.py ./
COPY --from=frontend-builder /app/dist ./frontend/dist
RUN mkdir -p /workspace/data /workspace/logs /workspace/frontend

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl --max-time 5 -fsS "http://127.0.0.1:${PORT:-7860}/healthz" || exit 1

CMD ["sh", "-c", "python -m uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-7860} --workers ${WORKERS:-1}"]
