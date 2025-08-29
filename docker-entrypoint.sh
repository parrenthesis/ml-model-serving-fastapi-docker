#!/usr/bin/env sh
set -euo pipefail

# Ensure a sane default for workers if not provided
: "${WEB_CONCURRENCY:=2}"

# BLAS/numexpr thread caps for predictable latency (overridable via env)
: "${OMP_NUM_THREADS:=1}"
: "${OPENBLAS_NUM_THREADS:=1}"
: "${MKL_NUM_THREADS:=1}"
: "${NUMEXPR_NUM_THREADS:=1}"

exec gunicorn \
  -k uvicorn.workers.UvicornWorker \
  -w "${WEB_CONCURRENCY}" \
  -b 0.0.0.0:8000 \
  app.main:app


