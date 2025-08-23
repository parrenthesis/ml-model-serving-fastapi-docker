# Design:
# - Load artifacts at startup for performance and stable schema enforcement.
# - Align inputs to model_features.json to prevent schema drift.
# - Provide both full-schema and minimal-schema endpoints to fit client workflows.
# - Enrich demographics on the server to reduce client complexity and errors
#   (clients send zipcode; we join demographics here).
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pickle

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
DATA_DIR = Path(os.getenv("DATA_DIR", str(ROOT_DIR / "data")))
MODEL_DIR = Path(os.getenv("MODEL_DIR", str(ROOT_DIR / "model")))

app = FastAPI(title="Real Estate Price Prediction API", version="1.0.0")

MODEL: Any = None
MODEL_FEATURES: List[str] = []
DEMOGRAPHICS: Optional[pd.DataFrame] = None
METRICS: Optional[Dict[str, Any]] = None

# Minimal sales fields expected from clients when using /predict_minimal.
# Any additional sales fields sent to /predict are accepted but ignored
# unless they exist in model_features.json.
SALE_FEATURES_REQUIRED: List[str] = [
	"bedrooms",
	"bathrooms",
	"sqft_living",
	"sqft_lot",
	"floors",
	"sqft_above",
	"sqft_basement",
	"zipcode",
]

class PredictResponse(BaseModel):
	"""Single prediction payload with optional metadata for client transparency."""
	prediction: float
	metadata: Dict[str, Any]


class BatchPredictResponse(BaseModel):
	"""Batch container for predictions.

	Using a batch wrapper provides a consistent response structure whether a
	single record or multiple records are submitted, and allows efficient
	bulk scoring without N round-trips.
	"""
	predictions: List[PredictResponse]


def _load_pickle(path: Path):
	with open(path, "rb") as f:
		return pickle.load(f)


def load_artifacts(model_dir: Path, data_dir: Path) -> Tuple[Any, List[str], pd.DataFrame]:
	"""Load model, feature list, and demographics table from disk."""
	model_path = model_dir / "model.pkl"
	features_path = model_dir / "model_features.json"
	demographics_path = data_dir / "zipcode_demographics.csv"

	if not model_path.exists() or not features_path.exists():
		raise FileNotFoundError(
			f"Model artifacts not found in {model_dir}. Run 'python create_model.py' first.")

	model = _load_pickle(model_path)
	model_features: List[str] = json.load(open(features_path))

	demographics = pd.read_csv(demographics_path, dtype={"zipcode": str})
	demographics = demographics.set_index("zipcode", drop=True)

	return model, model_features, demographics


@app.on_event("startup")
def startup_event() -> None:
	"""Load heavy artifacts once at process start to minimize per-request latency."""
	global MODEL, MODEL_FEATURES, DEMOGRAPHICS, METRICS
	MODEL, MODEL_FEATURES, DEMOGRAPHICS = load_artifacts(MODEL_DIR, DATA_DIR)
	metrics_path = MODEL_DIR / "metrics.json"
	if metrics_path.exists():
		try:
			METRICS = json.load(open(metrics_path))
		except Exception:
			METRICS = None


@app.get("/healthz")
def healthz() -> Dict[str, Any]:
	"""Liveness probe with lightweight model metadata for diagnostics."""
	return {
		"status": "ok",
		"num_features": len(MODEL_FEATURES) if MODEL_FEATURES else 0,
		"algorithm": (METRICS or {}).get("algorithm"),
	}


@app.get("/readyz")
def readyz() -> Dict[str, Any]:
	"""Readiness probe indicating whether artifacts are loaded and usable."""
	return {
		"model_loaded": MODEL is not None,
		"features_loaded": bool(MODEL_FEATURES),
		"demographics_loaded": DEMOGRAPHICS is not None,
	}


def _ensure_required_fields(df: pd.DataFrame, required: List[str]) -> None:
	"""Validate required fields are present; raise 422 with details if not."""
	missing = [c for c in required if c not in df.columns]
	if missing:
		raise HTTPException(status_code=422, detail={
			"error": "Missing required fields",
			"missing": missing
		})


def _enrich_with_demographics(df: pd.DataFrame, demographics: pd.DataFrame) -> pd.DataFrame:
	"""Join zipcode demographics and drop the zipcode column post-join.

	Raises 422 if zipcode is missing or unknown.
	"""
	if "zipcode" not in df.columns:
		raise HTTPException(status_code=422, detail="Field 'zipcode' is required for enrichment")
	df = df.copy()
	df["zipcode"] = df["zipcode"].astype(str)
	merged = df.merge(demographics.reset_index(), how="left", on="zipcode")
	# Only consider missing values in demographic columns for zipcode validation
	dem_cols = [c for c in demographics.reset_index().columns if c != "zipcode"]
	if merged[dem_cols].isna().any(axis=1).any():
		unknown = merged.loc[merged[dem_cols].isna().any(axis=1), "zipcode"].astype(str).unique().tolist()
		raise HTTPException(status_code=422, detail={
			"error": "Unknown zipcode(s) for demographics enrichment",
			"zipcodes": unknown
		})
	merged = merged.drop(columns=["zipcode"])  # zipcode not used by model after join
	return merged


def _align_to_model_features(df: pd.DataFrame, model_features: List[str]) -> pd.DataFrame:
	"""Filter and order columns to exactly match the training schema.

	Any extra columns provided by clients are dropped here, and missing columns
	are reported in a 422 error.
	"""
	missing = [c for c in model_features if c not in df.columns]
	if missing:
		raise HTTPException(status_code=422, detail={
			"error": "Missing features after enrichment",
			"missing": missing
		})
	# Keep only model features and order columns
	return df[model_features]


def _predict_dataframe(df: pd.DataFrame) -> List[float]:
	"""Run model inference and cast to JSON-serializable floats."""
	preds = MODEL.predict(df)
	return [float(x) for x in preds]


@app.post("/predict", response_model=BatchPredictResponse)
async def predict(payload: Union[Dict[str, Any], List[Dict[str, Any]]]) -> BatchPredictResponse:
	"""Predict from full-schema payload(s).

	Accept either a single JSON object or a list. Clients may send the entire
	future_unseen_examples.csv shape; extra fields are tolerated and dropped
	during alignment to model_features.json.
	"""
	records: List[Dict[str, Any]] = payload if isinstance(payload, list) else [payload]
	df = pd.DataFrame.from_records(records)

	_ensure_required_fields(df, required=["zipcode"])  # at minimum need zipcode to enrich
	enriched = _enrich_with_demographics(df, DEMOGRAPHICS)
	aligned = _align_to_model_features(enriched, MODEL_FEATURES)

	preds = _predict_dataframe(aligned)

	responses: List[PredictResponse] = []
	for pred in preds:
		meta = {
			"model_features": MODEL_FEATURES,
			"num_features": len(MODEL_FEATURES)
		}
		responses.append(PredictResponse(prediction=pred, metadata=meta))

	return BatchPredictResponse(predictions=responses)


@app.post("/predict_minimal", response_model=BatchPredictResponse)
async def predict_minimal(payload: Union[Dict[str, Any], List[Dict[str, Any]]]) -> BatchPredictResponse:
	"""Predict from a minimal set of sales fields; server fills demographics.

	This endpoint is optimized for manual entry or constrained clients. It
	reaches the same final feature vector via demographics join + alignment.
	"""
	records: List[Dict[str, Any]] = payload if isinstance(payload, list) else [payload]
	df = pd.DataFrame.from_records(records)

	_ensure_required_fields(df, required=SALE_FEATURES_REQUIRED)

	df_sale_only = df[SALE_FEATURES_REQUIRED]
	enriched = _enrich_with_demographics(df_sale_only, DEMOGRAPHICS)
	aligned = _align_to_model_features(enriched, MODEL_FEATURES)

	preds = _predict_dataframe(aligned)

	responses: List[PredictResponse] = []
	for pred in preds:
		meta = {
			"model_features": MODEL_FEATURES,
			"num_features": len(MODEL_FEATURES)
		}
		responses.append(PredictResponse(prediction=pred, metadata=meta))

	return BatchPredictResponse(predictions=responses)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
	"""Return a generic 500 error; avoid leaking internals to clients."""
	return JSONResponse(status_code=500, content={"error": "Internal Server Error"})
