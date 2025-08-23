# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1 \
	PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
	build-essential \
	&& rm -rf /var/lib/apt/lists/*

# Copy only requirement first for caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and data
COPY app ./app
COPY data ./data

# Copy pre-generated model artifacts (train before building the image)
# If model/ is absent, build will still succeed but API will fail readiness until
# artifacts are provided at runtime (e.g., mounted volume or remote fetch).
COPY model ./model

ENV MODEL_DIR=/app/model \
	DATA_DIR=/app/data

EXPOSE 8000
# For demos only: to train inside the container, you could replace CMD with
# `python create_model.py && uvicorn app.main:app --host 0.0.0.0 --port 8000`
# but best practice is to train before build and keep images immutable.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
