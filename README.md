# ML Model Serving – FastAPI + Docker

Example real estate price prediction API. Trains a simple model and serves predictions via FastAPI, packaged with Docker. Includes server-side demographics enrichment, health/readiness probes, confidence intervals, and a metrics endpoint.

## Quickstart

Prereqs (either path works):
- Local dev: Poetry (Python 3.11)
- Container: Docker + docker compose

### One-liners (Docker)
```bash
docker compose up --build -d
curl -sSf http://localhost:8000/healthz && echo && \
  curl -sSf http://localhost:8000/readyz && echo && \
  curl -sSf http://localhost:8000/metrics
```
Stop:
```bash
docker compose down
```

### Local (Poetry)
```bash
poetry install --no-root
poetry run python create_model.py      # train (KNN with confidence intervals)
# run with gunicorn/uvicorn worker (production-like)
gunicorn -k uvicorn.workers.UvicornWorker -w ${WEB_CONCURRENCY:-2} -b 0.0.0.0:8000 app.main:app
```
Smoke test:
```bash
poetry run python tests/test_api.py
```

## Endpoints
- POST `/predict` (full schema)
  - JSON matches `data/future_unseen_examples.csv` (no price/date/id). Server joins demographics by `zipcode` and aligns to training features.
  - Accepts a single object or a list (batch). When a list is provided, `MAX_BATCH` is enforced (422 if exceeded).
  - **Returns confidence intervals**: prediction ranges, confidence scores, and feature novelty metrics.
- POST `/predict_minimal` (minimal schema)
  - Only the following fields are required; the server fills demographics:
    - `bedrooms,bathrooms,sqft_living,sqft_lot,floors,sqft_above,sqft_basement,zipcode`
  - Accepts a single object or a list (batch). `MAX_BATCH` enforced for lists.
  - **Returns confidence intervals**: prediction ranges, confidence scores, and feature novelty metrics.
- GET `/healthz` – liveness
- GET `/readyz` – readiness (artifacts loaded)
- GET `/metrics` – model/service metrics and metadata (no inference)
- GET `/metrics_prom` – Prometheus exposition (enabled via env)

Minimal example:
```bash
curl -s -X POST http://localhost:8000/predict_minimal \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: <your-key>' \
  -d '{
    "bedrooms": 3, "bathrooms": 2.0, "sqft_living": 2000,
    "sqft_lot": 5000, "floors": 1.0, "sqft_above": 1500,
    "sqft_basement": 500, "zipcode": "98118"
  }'
```

Response includes confidence intervals:
```json
{
  "prediction": 450000,
  "confidence_interval": {"lower": 420000, "upper": 480000},
  "confidence_score": 0.85,
  "confidence_type": "hybrid",
  "feature_novelty": 0.12
}
```

## Training
- Run `python create_model.py` to produce `model/`:
  - `model.pkl` (pipeline), `model_features.json` (training schema), `metrics.json` (R², RMSE, MAE, Median AE, sizes; includes tuning info when used)
  - **Confidence artifacts**: `quantile_*.pkl` (XGBoost), `training_data.pkl` (both models)
- Algorithms:
  - KNN with `RobustScaler` (default): `--algo knn`
  - XGBoost (included): `--algo xgboost`
- **Confidence intervals enabled by default** (`--confidence` flag)
- Tuning (cross‑validation):
  - KNN tuned: `poetry run python create_model.py --algo knn --tune knn --cv-folds 5`
  - XGB tuned: `poetry run python create_model.py --algo xgboost --tune xgb --cv-folds 5 --n-iter 30`
- CPU limiting: set `--max-workers 2` or env `MAX_WORKERS=2` to cap parallelism.

## Configuration
- `MODEL_DIR` (default `/app/model`) – where artifacts are loaded from
- `DATA_DIR` (default `/app/data`) – where `zipcode_demographics.csv` is read from
- **`CONFIDENCE_ENABLED`** (default `true`) – enable confidence intervals
- **`CONFIDENCE_METHOD`** (default `hybrid`) – `quantile`, `knn_variance`, `feature_distance`, or `hybrid`
- `INPUT_EXTRA_POLICY` – how `/predict` handles unknown fields
  - `allow` (default), `ignore`, or `forbid`
  - Set in compose (default: allow) or as an env var locally
- `API_KEYS` – optional comma-separated list. If set, `X-API-Key` is required on predict endpoints.
- `RATE_LIMIT_PER_MINUTE` – optional integer per-identity limit; 0 disables.
- `MODEL_VERSION` – optional override; also inferred from `metrics.json` if present.
- `WEB_CONCURRENCY` – gunicorn worker count (default 2).
- Zipcode validated as 5 digits (regex)
- `MAX_BATCH` – maximum list size for batch requests (default 512).
- `PROMETHEUS_ENABLED` – enable `/metrics_prom` exposition (default false).
- `LOG_JSON` – enable structured JSON logs; `REQUEST_ID_HEADER` can supply or propagate request id.
- Hybrid artifacts (optional; falls back to local on failure):
  - `MODEL_SOURCE` – `local|http|s3` (default `local`).
  - `MODEL_URL` – base URL for HTTP when `MODEL_SOURCE=http`.
  - `MODEL_S3_URI` – `s3://bucket/prefix` when `MODEL_SOURCE=s3`.
  - `MODEL_SHA256` – optional checksum for `model.pkl`; verify when provided.

Notes
- If `API_KEYS` is set, include `-H 'X-API-Key: <your-key>'` on requests.
- `model_version` appears in `/healthz` and `/metrics` for visibility.
- **Confidence intervals add ~2-3ms latency** while providing comprehensive uncertainty quantification.

## Dev commands (Makefile)
```bash
make train        # train KNN
make train-xgb    # train XGBoost
make train-knn-tuned    # train KNN with CV tuning
make train-xgb-tuned    # train XGB with CV tuning
make api          # run FastAPI locally (gunicorn)
make test-api     # post sample rows to /predict
make docker-build # build image (multi-stage)
make up           # compose up (builds and starts)
make down         # compose down
make health       # curl health & ready
make ci-smoke     # build, run, health checks, stop
```

## Data notes
- `data/zipcode_demographics.csv` and `data/future_unseen_examples.csv` are included for runtime/tests.
- `data/kc_house_data.csv` (training) is excluded from VCS; artifacts are pre-generated in `model/`.

## Architecture
- **Multi-stage Docker build**: Builder stage installs dependencies, runtime stage copies only what's needed
- FastAPI + Pydantic for performance and strict typing/docs
- Server-side enrichment for consistency and governance
- **Confidence intervals**: Hybrid method combining quantile regression, KNN variance, and feature distance
- Docker for reproducible deploys; compose for local/PoC runs
- Separate training from build (immutable images; faster CI)
- Optional Prometheus metrics and structured logs for observability; batch support for throughput

## License
MIT (code). Respect any data source restrictions.
