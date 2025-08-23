# ML Model Serving – FastAPI + Docker

Example real estate price prediction API. Trains a simple model and serves predictions via FastAPI, packaged with Docker. Includes server-side demographics enrichment, health/readiness probes, and a metrics endpoint.

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
poetry run python create_model.py      # train (KNN)
poetry run uvicorn app.main:app --host 0.0.0.0 --port 8000
```
Smoke test:
```bash
poetry run python tests/test_api.py
```

## Endpoints
- POST `/predict` (full schema)
  - JSON matches `data/future_unseen_examples.csv` (no price/date/id). Server joins demographics by `zipcode` and aligns to training features.
- POST `/predict_minimal` (minimal schema)
  - Only the following fields are required; the server fills demographics:
    - `bedrooms,bathrooms,sqft_living,sqft_lot,floors,sqft_above,sqft_basement,zipcode`
- GET `/healthz` – liveness
- GET `/readyz` – readiness (artifacts loaded)
- GET `/metrics` – model/service metrics and metadata (no inference)

Minimal example:
```bash
curl -s -X POST http://localhost:8000/predict_minimal \
  -H 'Content-Type: application/json' \
  -d '{
    "bedrooms": 3, "bathrooms": 2.0, "sqft_living": 2000,
    "sqft_lot": 5000, "floors": 1.0, "sqft_above": 1500,
    "sqft_basement": 500, "zipcode": "98118"
  }'
```

## Training
- Run `python create_model.py` to produce `model/`:
  - `model.pkl` (pipeline), `model_features.json` (training schema), `metrics.json` (R², RMSE, MAE, Median AE, sizes)
- Default algorithm: KNN with `RobustScaler`
- Optional: XGBoost with `--algo xgboost` (install `xgboost` first)

## Configuration
- `MODEL_DIR` (default `/app/model`) – where artifacts are loaded from
- `DATA_DIR` (default `/app/data`) – where `zipcode_demographics.csv` is read from
- `INPUT_EXTRA_POLICY` – how `/predict` handles unknown fields
  - `allow` (default), `ignore`, or `forbid`
  - Set in compose (default: allow) or as an env var locally
- Zipcode validated as 5 digits (regex)

## Dev commands (Makefile)
```bash
make train        # train KNN
make train-xgb    # train XGBoost (requires xgboost)
make api          # run FastAPI locally
make test-api     # post sample rows to /predict
make docker-build # build image
make up           # compose up (builds and starts)
make down         # compose down
make health       # curl health & ready
make ci-smoke     # build, run, health checks, stop
```

## Data notes
- `data/zipcode_demographics.csv` and `data/future_unseen_examples.csv` are included for runtime/tests.
- `data/kc_house_data.csv` (training) is excluded from VCS; artifacts are pre-generated in `model/`.

## Why these choices
- FastAPI + Pydantic for performance and strict typing/docs
- Server-side enrichment for consistency and governance
- Docker for reproducible deploys; compose for local/PoC runs
- Separate training from build (immutable images; faster CI)

## License
MIT (code). Respect any data source restrictions.
