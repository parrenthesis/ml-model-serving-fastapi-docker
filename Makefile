.PHONY: train train-xgb api test-api docker-build up down health ci-smoke help

PY=poetry run

train: ## Train KNN model and write artifacts to model/
	$(PY) python create_model.py

train-knn-tuned: ## Train KNN with CV tuning and write artifacts
	$(PY) python create_model.py --algo knn --tune knn --cv-folds 5

train-xgb: ## Train XGBoost model (requires xgboost) and write artifacts
	$(PY) python create_model.py --algo xgboost

train-xgb-tuned: ## Train XGBoost with CV tuning and write artifacts
	$(PY) python create_model.py --algo xgboost --tune xgb --cv-folds 5 --n-iter 30

api: ## Run FastAPI locally with gunicorn/uvicorn worker
	gunicorn -k uvicorn.workers.UvicornWorker -w $${WEB_CONCURRENCY:-2} -b 0.0.0.0:8000 app.main:app

test-api: ## Post samples from future_unseen_examples.csv to /predict
	$(PY) python tests/test_api.py

docker-build: ## Build Docker image locally
	docker build -t housing-api:latest .

up: ## Compose up (build and start container)
	docker compose up --build -d

down: ## Compose down (stop/remove container)
	docker compose down

health: ## Check healthz and readyz endpoints
	curl -sf http://localhost:8000/healthz && echo && curl -sf http://localhost:8000/readyz

ci-smoke: docker-build ## Build, run, health check, and stop
	docker run -d -p 8000:8000 --name api housing-api:latest
	sleep 3 && curl -sf http://localhost:8000/healthz && curl -sf http://localhost:8000/readyz
	docker rm -f api || true

help: ## Show make targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' Makefile | sort | awk 'BEGIN {FS=":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'
