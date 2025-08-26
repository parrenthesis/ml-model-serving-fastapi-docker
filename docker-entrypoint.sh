#!/usr/bin/env sh
set -euo pipefail

# Ensure a sane default for workers if not provided
: "${WEB_CONCURRENCY:=2}"

exec gunicorn \
  -k uvicorn.workers.UvicornWorker \
  -w "${WEB_CONCURRENCY}" \
  -b 0.0.0.0:8000 \
  app.main:app


