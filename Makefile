PY=poetry run

train:
	$(PY) python create_model.py

train-xgb:
	$(PY) python create_model.py --algo xgboost

api:
	$(PY) uvicorn app.main:app --host 0.0.0.0 --port 8000

test-api:
	$(PY) python tests/test_api.py

docker-build:
	docker build -t housing-api:latest .

up:
	docker compose up --build -d

down:
	docker compose down

health:
	curl -sf http://localhost:8000/healthz && echo && curl -sf http://localhost:8000/readyz

ci-smoke: docker-build
	docker run -d -p 8000:8000 --name api housing-api:latest
	sleep 3 && curl -sf http://localhost:8000/healthz && curl -sf http://localhost:8000/readyz
	docker rm -f api || true
