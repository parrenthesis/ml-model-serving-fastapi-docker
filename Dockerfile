# syntax=docker/dockerfile:1
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1 \
	PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
	build-essential \
	&& rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.11-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1 \
	PIP_NO_CACHE_DIR=1
WORKDIR /app

# System deps (runtime only)
RUN apt-get update && apt-get install -y --no-install-recommends \
	&& rm -rf /var/lib/apt/lists/*

# Copy installed site-packages from builder
COPY --from=builder /install /usr/local

# Copy application code and data
COPY app ./app
COPY data ./data
# Copy all model artifacts including confidence models
COPY model ./model

ARG MODEL_VERSION=dev
ENV MODEL_VERSION=${MODEL_VERSION}
LABEL org.opencontainers.image.title="housing-api" \
      org.opencontainers.image.version="${MODEL_VERSION}"

ENV MODEL_DIR=/app/model \
	DATA_DIR=/app/data \
	WEB_CONCURRENCY=2 \
	RATE_LIMIT_PER_MINUTE=0

EXPOSE 8000
# For demos only: to train inside the container, replace CMD with
# `python create_model.py && gunicorn -k uvicorn.workers.UvicornWorker -w $WEB_CONCURRENCY -b 0.0.0.0:8000 app.main:app`
# Use shell form so env vars (WEB_CONCURRENCY) expand with defaults.
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh
CMD ["/usr/local/bin/docker-entrypoint.sh"]
